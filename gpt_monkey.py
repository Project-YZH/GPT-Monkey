import subprocess
import json
from lxml import etree
import requests
import tkinter as tk
from tkinter import messagebox
import uiautomator2 as u2  # uiautomator2 2.16.25
import os
import re
import openai  # openai 0.27.8
import numpy as np
import pandas as pd
import csv
from sklearn.metrics.pairwise import cosine_similarity


class OpenAIChatClient:
    def __init__(self):
        self.api_key = "sk-"  # API KEY
        self.proxies = {"http": "http://127.0.0.1:10809", "https": "http://127.0.0.1:10809"}
        self.url = 'https://api.openai.com/v1/chat/completions'
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }

    def send_request(self, prompt, img_url):
        data = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": img_url
                            }
                        }
                    ]
                }
            ]
        }
        # def send_request(self, content):
        #     data = {
        #         'model': 'gpt-3.5-turbo',
        #         'messages': [{'role': 'user', 'content': content}],
        #         'temperature': 0.7
        #     }
        json_data = json.dumps(data)
        print("GPT-Request:\n", json_data)
        response = requests.post(self.url, data=json_data, headers=self.headers, proxies=self.proxies)

        try:
            response_json = response.json()
            print("GPT-Response:\n", response_json)
            return response_json
        except ValueError:
            print("NOT JSON:", response.text)
            return {"error": "Invalid response format"}


client = OpenAIChatClient()


# *************


def get_uix2():
    d = u2.connect()
    xml = d.dump_hierarchy()
    root = etree.fromstring(xml.encode('utf-8'))
    nodes_list = []

    def dfs(element):
        if element.tag == "node":
            bounds = element.get("bounds")
            resource_id = element.get("resource-id", "")
            text = element.get("text", "")
            if bounds:
                nodes_list.append((bounds, resource_id, text))

        for child in element:
            dfs(child)

    dfs(root)

    simplified_xml = "<ui>"
    for bounds, resource_id, text in nodes_list:
        simplified_xml += f'<node bounds="{bounds}" resource-id="{resource_id}" text="{text}"/>'
    simplified_xml += "</ui>"

    return simplified_xml


def get_screenshot():
    d = u2.connect()

    screenshot_path = "screenshot.png"

    d.screenshot(screenshot_path)

    return screenshot_path


def get_current_activity():
    d = u2.connect()

    current_app_info = d.app_current()

    current_activity = current_app_info.get("activity")

    return current_activity


def get_current_package_name():
    try:

        d = u2.connect()

        current_package_name = d.app_current().get("package")
        return current_package_name
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def upload(path):
    headers = {'Authorization': 'KguGCPjhzM2z7lGDoRkjzXCMi1tYehNI'}  # sm.ms API
    files = {'smfile': open(path, 'rb')}
    url = 'https://sm.ms/api/v2/upload'

    res = requests.post(url, files=files, headers=headers).json()

    if res.get("success"):
        return res['data']['url']
    else:
        print("ERROR:", res.get("message"))
        return None


def get_interface_association():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    interface_association_path = os.path.join(current_dir, "interface_association.txt")

    try:
        with open(interface_association_path, "r", encoding="utf-8") as file:
            interface_association = file.read().strip()
    except FileNotFoundError:
        print("Interface association file not found. Using default value.")
        interface_association = "Default association text"
    return interface_association


global first_device, time_, between_test_app, duration, delay


def open_details_window(text_content):

    details_window = tk.Toplevel()
    details_window.title("GPT-Monkey Details 1")
    details_window.geometry("610x460")

    content_frame = tk.Frame(details_window)
    content_frame.pack(fill="both", expand=True)

    title_label = tk.Label(content_frame, text="GPT-Monkey", font=("Arial", 26))
    title_label.pack(pady=(20, 10))

    try:
        result = subprocess.run(["adb", "devices"], capture_output=True, text=True, check=True)
        output = result.stdout
        device_list = re.findall(r"(\S+)\s+device", output)

        Scene_preset = (
            "\nGPT-Monkey employs LLM to segment the targeted function including the entry area of the function and its associated interfaces, tailor testing parameters and decode the testing parameters into the executable scripts for Monkey, which can realize the testing directed to a specific function of the app. The image I sent you are for reference only, please analyze mainly my text.")

        # Get the path of the current Python file's directory
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Define the path to interface_association.txt in the same directory
        interface_association_path = os.path.join(current_dir, "interface_association.txt")

        # Read Interface_Association from the local text file
        try:
            with open(interface_association_path, "r", encoding="utf-8") as file:
                interface_association = file.read().strip()
        except FileNotFoundError:
            print("Interface association file not found. Using default value.")
            interface_association = "Default association text"

        if device_list:
            first_device = device_list[1]
            ui_hierarchy = get_uix2()

            screenshot_path = get_screenshot()
            screenshot_url = upload(screenshot_path)
            interface_association = "\ncurrent_activity:\n" + get_current_activity() + "\nglobal interface association:\n" + get_interface_association()

            # Initialize basic_parameters as an empty dictionary
            basic_parameters = {}

            basic_parameters["Device"] = f"{first_device} Connected"

            # Query for "Duration" parameter
            input_duration = (
                    Scene_preset
                    + "\nUser_Requirements:\n" + text_content
                    + "Interface Layout:\n" + ui_hierarchy
                    + "\nInterface Screenshot:\n" + screenshot_url
                    + "\nInterface Association" + interface_association
                    + "\nTask 1:"
                    + "\nGenerate the Duration parameter primarily from the testing requirements input by the user, while also referencing the interface, the interface screenshot and the global interface association. Only return '* minutes', no extra answers.The code for handling the response afterwards is:basic_parameters[Duration] = json.loads(response_content).get(Duration, Default Duration)."
                    + "\nTask 2:"
                    + "\nSpecify the Log Level setting for the test primarily from the testing requirements input by the user, while also referencing the interface layout and the global interface association. Only return 'simplified mode' or 'complicated mode', no extra answers."
                    + "\nTask 3:"
                    + "\nDetermine the appropriate Output Path for storing log files primarily from the testing requirements input by the user. For example, only return '/sdcard/monkey_output', no extra answers."
                    + "\nTask 4:"
                    + "\nIndicate whether segmentation of the function is needed primarily from the User requirements input by the user. If a function is mentioned in the User requirements, function segmentation is required. Only return 'Yes' or 'No', no extra answers."
                    + "\nTask 5:"
                    + "\nSpecify the Target Function of the test primarily from the testing requirements input by the user, while also referencing the interface layout and the global interface association. Only return the 'Target Function: name of the target function', no extra answers."
                    + "\nTask 6:"
                    + "\nList only one test parameter that Monkey can directly support in addition to the above but are still necessary primarily from the testing requirements input by the user, while also referencing the interface layout and the global interface association. If this parameter is necessary, please give a specific value for the parameter. If it is not necessary, you can return None. For example, only return 'Other Parameters: --throttle 1000 --pct-touch 30 --pct-motion 20 --pct-trackball 10 --pct-anyevent 25', no extra answers."

            )
            print("\nInput to ChatGPT for Duration: " + input_duration)
            # result_duration = client.send_request(input_duration)
            result_duration = client.send_request(input_duration, screenshot_url)
            if result_duration and "choices" in result_duration:
                response_content = result_duration["choices"][0]["message"]["content"]

                duration_match = re.search(r"(\d+ minutes)", response_content)
                basic_parameters["Duration"] = duration_match.group(0)

                log_level_match = re.search(r"(simplified mode|complicated mode)", response_content)
                basic_parameters["Log Level"] = log_level_match.group(0) if log_level_match else "Simplified mode"

                output_path_match = re.search(r"/sdcard/(.+)", response_content)
                basic_parameters["Output Path"] = output_path_match.group(
                    0) if output_path_match else "/sdcard/monkey_output"

                segment_function_match = re.search(r"(Yes|No)", response_content)
                basic_parameters["Segment Function?"] = segment_function_match.group(
                    0) if segment_function_match else "No"

                target_match = re.search(r"Target Function: (.+)", response_content)
                basic_parameters["Target"] = target_match.group(1) if target_match else "None"

                other_parameters_match = re.search(r"Other Parameters: (.+)", response_content, re.DOTALL)
                basic_parameters["Other Parameters"] = other_parameters_match.group(
                    1) if other_parameters_match else "None"

                print(basic_parameters)

        else:
            first_device = "No device found"
            print("No device found.")
            # Default values when no device is found
            basic_parameters = {
                "Device": "No device found",
                "Duration": duration,
                "Log Level": "Simplified mode",
                "Output Path": "C:/Users/YZH/Desktop/log/output/",
                "Segment Function?": "No",
                "Target": "None",
                "Other Parameters": "None"
            }

    except Exception as e:
        print(f"Error: {e}")
        basic_parameters = {
            "Device": "No device found",
            "Duration": "100 ms",
            "Log Level": "Simplified mode",
            "Output Path": "C:/Users/YZH/Desktop/log/output/",
            "Segment Function?": "No",
            "Target": "None",
            "Other Parameters": "None"
        }

    info_frame = tk.Frame(content_frame)
    info_frame.pack(pady=10)

    entries = []
    for i, (label_text, entry_text) in enumerate([
        ("Device", basic_parameters["Device"]),
        ("Duration", basic_parameters["Duration"]),
        ("Log Level", basic_parameters["Log Level"]),
        ("Output Path", basic_parameters["Output Path"]),
        ("Segment Function?", basic_parameters["Segment Function?"]),
        ("Target", basic_parameters["Target"]),
        ("Other Parameters", basic_parameters["Other Parameters"])
    ]):
        row_frame = tk.Frame(info_frame)
        row_frame.pack(fill="x", padx=20, pady=5)

        label = tk.Label(row_frame, text=f"{label_text}:", font=("Arial", 16), anchor="e", width=16)
        label.pack(side="left")

        entry = tk.Entry(row_frame, font=("Arial", 16), width=28)
        entry.pack(side="left", padx=10)
        entry.insert(0, entry_text)
        entries.append(entry)

    button_frame = tk.Frame(details_window)
    button_frame.pack(side="bottom", fill="x", pady=20)

    def get_entries():

        info = {
            "Device": entries[0].get(),
            "Duration": entries[1].get(),
            "Log Level": entries[2].get(),
            "Output Path": entries[3].get(),
            "Segment Function?": entries[4].get(),
            "Target": entries[5].get(),
            "Other Parameters": entries[6].get()
        }

        current_package_name = get_current_package_name()

        new_prompt = (
                Scene_preset
                + f"\nBasic Parameters:\nDevice: {info['Device']}\nDuration: {info['Duration']}\nLog Level: {info['Log Level']}\n"
                  f"Output Path: {info['Output Path']}\nSegment Function?: {info['Segment Function?']}\nTarget: {info['Target']}\n"
                  f"Other Parameters: {info['Other Parameters']}"
                + f"\nCurrent Package Name:\n{current_package_name}"
                + f"\nInterface Layout:\n{ui_hierarchy}\nInterface Screenshot:\n{screenshot_url}\nInterface Association:\n{interface_association}\n"
                + f"Task 1:\n"
                + f"Extract the coordinates of the target function's entry area in the current interface by analyzing the target funcetion and the interface layout text. Make sure that the text that matches the target function is matched in the layout node, and then return the bounds of the node. Return only the 'Entry area coordinates: [**,**][**,**]' format, without any extra responses. "
                + f"\nTask 2:\n"
                + f"Segment the interface associated with the target function primarily from the Interface Association, while also referencing the basic Parameters, the interface layout and the interface screenshot. The result of segmenting the associated interface is to segment the activities that can be connected to the current activity through edges. The current activity is already clearly written in the Interface Association, and other activities are in its node information. Return only the 'Associated interfaces: .** .**' format, without any extra responses. Do not use 'n' for line breaks."
                + f"\nTask 3:\n"
                + f"Generate test instruction that can directly start Monkey for the target function from the basic Parameters, the target function, the interface layout, the interface screenshot and the global interface association. The parameters in the instruction should include all the parameters mentioned in the basic parameters and the parameters you suggest. Return only the 'Instruction: adb shell CLASSPATH=/sdcard/monkey. jar:/sdcard/framework. jar exec app_process /system/bin tv.panda.test.monkey.Monkey -p com.ichi2.anki --uiautomatormix --running-minutes 2 --output-directory /sdcard/monkey_output' format. Some parameters need to be added but don't add simple mode. com.ichi2.ankicom.ichi2.anki needs to be replaced with Current Package Name. Without any extra responses."
        )

        openai.api_key = "sk-c-"  # API KEY

        def generate_embedding(text):
            try:
                response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
                embedding = response['data'][0]['embedding']

                return embedding
            except Exception as e:
                print(f"ERROR：{e}")
                return None

        def txt_to_embeddings_csv(input_txt_file, output_csv_file):
            try:

                with open(input_txt_file, "r", encoding="utf-8") as file:
                    lines = file.readlines()

                embeddings = []
                for line in lines:
                    line = line.strip()
                    if line:
                        embedding = generate_embedding(line)
                        embeddings.append(embedding)

                with open(output_csv_file, "w", newline='', encoding="utf-8") as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(["Text", "Embedding"])
                    for i, line in enumerate(lines):
                        csv_writer.writerow([line, embeddings[i]])

                print(f"Embedding successfully saved to {output_csv_file}")

            except Exception as e:
                print(f"ERROR：{e}")

        input_txt_file = "Knowledge Document.txt"
        output_csv_file = "text_embeddings.csv"
        txt_to_embeddings_csv(input_txt_file, output_csv_file)

        def load_embeddings_from_csv(file_path):
            df = pd.read_csv(file_path)

            df['Embedding'] = df['Embedding'].fillna("[]")

            df['Embedding'] = df['Embedding'].apply(lambda x: json.loads(x) if isinstance(x, str) else [])
            return df

        embedding_df = load_embeddings_from_csv('text_embeddings.csv')

        def generate_query_embedding(query_text):
            response = openai.Embedding.create(
                input=query_text,
                model="text-embedding-ada-002"
            )
            embedding = response['data'][0]['embedding']
            return embedding

        query_embedding = generate_query_embedding(new_prompt)

        def find_most_similar_texts(query_embedding, embedding_df, top_k=5):

            embeddings = np.vstack(embedding_df['Embedding'].values)
            query_embedding = np.array(query_embedding).reshape(1, -1)
            similarities = cosine_similarity(query_embedding, embeddings).flatten()

            top_k_indices = similarities.argsort()[-top_k:][::-1]
            top_texts = embedding_df.iloc[top_k_indices]['Text'].tolist()
            return top_texts

        related_texts = find_most_similar_texts(query_embedding, embedding_df)
        print("Top related texts:", related_texts)

        # def generate_answer(query_text, related_texts):
        #
        #     # Combine related texts into a single context string
        #     context = "\n".join(related_texts)
        #
        #     response = openai.ChatCompletion.create(
        #         model="gpt-3.5-turbo",
        #         messages=[
        #             {"role": "system", "content": "You are a helpful assistant."},
        #             {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query_text}"}
        #         ],
        #         temperature=0.7
        #     )
        #
        #     # Extract the answer from the response
        #     answer = response.choices[0].message['content'].strip()
        #     return answer

        def generate_answer(query_text, related_texts, img_url):
            # Combine related texts into a single context string
            context = "\n".join(related_texts)

            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Context:\n{context}\n\nQuestion: {query_text}"
                            }
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": img_url
                                }
                            }
                        ]
                    }
                ],
                temperature=0.7
            )

            # Extract the answer from the response
            answer = response.choices[0].message['content'].strip()
            return answer

        print("\nNew Prompt:\n" + new_prompt)

        response = generate_answer(new_prompt, related_texts, screenshot_url)
        print(response)
        response_content = response

        entry_area_match = re.search(r"\[\d+,\d+\]\[\d+,\d+\]", response_content)
        entry_area_coordinates = entry_area_match.group(0) if entry_area_match else "No coordinates found"

        associated_interfaces_match = re.search(
            r"Associated interfaces:\s*(.+?)\n(?:Task 3|Instruction:)", response_content, re.DOTALL)
        associated_interfaces_1 = associated_interfaces_match.group(
            1).strip() if associated_interfaces_match else "No activities found"

        package_name = get_current_package_name()
        if package_name is None:
            package_name = "unknown.package"  # Fallback if package name retrieval fails

        modified_interfaces = []
        for line in associated_interfaces_1.split():
            line = line.strip()
            if line.startswith("."):
                # Add full package name for activities starting with a dot
                modified_interfaces.append(f"{package_name}{line}")
            else:
                # Use the activity name as is
                modified_interfaces.append(line)

        # Join the modified interfaces back into a single string if needed
        associated_interfaces = "\n".join(modified_interfaces)

        instruction_match = re.search(r"adb shell .+", response_content)
        final_instruction = instruction_match.group(
            0) + " --act-whitelist-file /sdcard/awl.strings" if instruction_match else "No instruction found"

        # response = client.send_request(new_prompt)

        show_final_command_window(entry_area_coordinates, associated_interfaces, final_instruction)

    confirm_button = tk.Button(button_frame, text="Confirm", command=get_entries, font=("Arial", 16))
    confirm_button.pack(side="right", padx=(10, 20))

    cancel_button = tk.Button(button_frame, text="Cancel", command=details_window.destroy, font=("Arial", 16))
    cancel_button.pack(side="right", padx=(10, 10))


# *************


def check_and_push_files():
    adb_command_monkey = "adb shell ls /sdcard/monkey.jar"
    adb_command_framework = "adb shell ls /sdcard/framework.jar"

    try:
        subprocess.run(adb_command_monkey, shell=True, check=True)
        print("/sdcard/monkey.jar File exists")
    except subprocess.CalledProcessError:
        print("/sdcard/monkey.jar The file does not exist, the push command will be executed")
        subprocess.run("adb push monkey.jar /sdcard", shell=True, check=True)

    try:
        subprocess.run(adb_command_framework, shell=True, check=True)
        print("/sdcard/framework.jar File exists")
    except subprocess.CalledProcessError:
        print("/sdcard/framework.jar The file does not exist, the push command will be executed")
        subprocess.run("adb push framework.jar /sdcard", shell=True, check=True)


def remove_area(screen_width, screen_height, remove_bounds):
    top_area = [0, 0, screen_width, remove_bounds[1]]
    bottom_area = [0, remove_bounds[3], screen_width, screen_height]
    left_area = [0, remove_bounds[1], remove_bounds[0], remove_bounds[3]]
    right_area = [remove_bounds[2], remove_bounds[1], screen_width, remove_bounds[3]]

    top_area_str = f"[{top_area[0]},{top_area[1]}][{top_area[2]},{top_area[3]}]"
    bottom_area_str = f"[{bottom_area[0]},{bottom_area[1]}][{bottom_area[2]},{bottom_area[3]}]"
    left_area_str = f"[{left_area[0]},{left_area[1]}][{left_area[2]},{left_area[3]}]"
    right_area_str = f"[{right_area[0]},{right_area[1]}][{right_area[2]},{right_area[3]}]"

    return [top_area_str, bottom_area_str, left_area_str, right_area_str]

    # data = [{"bounds": top_area_str}, {"bounds": bottom_area_str}, {"bounds": left_area_str}, {"bounds": right_area_str}]
    # json_str = json.dumps(data, indent=4)
    # return json_str


def parse_coordinates(input_str):
    numbers = re.findall(r'\d+', input_str)
    coordinates = [int(num) for num in numbers]
    return coordinates


def run_adb(adb_command):
    try:
        subprocess.run(adb_command, shell=True, check=True)
        print("Target Monkey test successfully")
    except subprocess.CalledProcessError as e:
        print(f"Target Monkey test successfully：{e}")


def write_and_push_file(file_content, local_file_name, remote_file_name):
    current_directory = os.getcwd()
    temp_dir = os.path.join(current_directory, "temp")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    local_file_full_path = os.path.join(temp_dir, local_file_name)

    with open(local_file_full_path, "w") as file:
        file.write(file_content)

    remote_file_path = f"/sdcard/{remote_file_name}"
    adb_push_command = f"adb push {local_file_full_path} {remote_file_path}"

    try:
        subprocess.run(adb_push_command, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error pushing file {local_file_name}: {e}")
        return False


def show_final_command_window(test_free_area, activity_list, final_results):
    result_window = tk.Toplevel()
    result_window.title("GPT-Monkey Details 2")
    result_window.geometry("610x460")

    test_free_label = tk.Label(result_window, text="Targeted function entry area coordinates:", font=("Arial", 16))
    test_free_label.pack(anchor="w", padx=10, pady=(10, 0))

    test_free_text = tk.Text(result_window, height=2, width=60, font=("Arial", 16))
    test_free_text.pack(padx=10, pady=(0, 10))
    test_free_text.insert("1.0", test_free_area)

    activity_label = tk.Label(result_window, text="Targeted function associated interfaces' activities:",
                              font=("Arial", 16))
    activity_label.pack(anchor="w", padx=10, pady=(10, 0))

    activity_text = tk.Text(result_window, height=4, width=60, font=("Arial", 16))
    activity_text.pack(padx=10, pady=(0, 10))
    activity_text.insert("1.0", activity_list)

    final_command_label = tk.Label(result_window, text="Testing instruction:", font=("Arial", 16))
    final_command_label.pack(anchor="w", padx=10, pady=(10, 0))

    final_command_text = tk.Text(result_window, height=4, width=60, font=("Arial", 16))
    final_command_text.pack(padx=10, pady=(0, 10))
    final_command_text.insert("1.0", final_results)

    buttons_frame = tk.Frame(result_window)
    buttons_frame.pack(side="bottom", anchor="e", padx=10, pady=10)

    def send_instruction():

        test_free_area_input = test_free_text.get("1.0", tk.END).strip()
        activity_list_input = activity_text.get("1.0", tk.END).strip()

        remove_bounds = parse_coordinates(test_free_area_input)
        if len(remove_bounds) != 4:
            messagebox.showerror("Error", "Please enter valid coordinates.")
            return

        screen_width, screen_height = get_screen_dimensions()

        test_free_areas_bounds_list = remove_area(screen_width, screen_height, remove_bounds)

        current_activity = get_current_package_name() + get_current_activity()
        if not current_activity:
            messagebox.showerror("Error", "Failed to get current activity.")
            return

        # max.widget.black
        max_widget_black_entries = []
        for area_bounds in test_free_areas_bounds_list:
            entry = {
                "activity": current_activity,
                "bounds": area_bounds
            }
            max_widget_black_entries.append(entry)

        max_widget_black_content = json.dumps(max_widget_black_entries, indent=4)

        #  max.widget.black
        success = write_and_push_file(max_widget_black_content, 'max.widget.black', 'max.widget.black')
        if not success:
            messagebox.showerror("Error", "Failed to push max.widget.black.")
            return

        # awl.strings
        awl_strings_content = activity_list_input.strip()

        # awl.strings
        success = write_and_push_file(awl_strings_content, 'awl.strings', 'awl.strings')
        if not success:
            messagebox.showerror("Error", "Failed to push awl.strings.")
            return

        check_and_push_files()
        d.uiautomator.stop()
        # d.stop_uiautomator()

        final_command = final_command_text.get("1.0", tk.END).strip()
        if not final_command:
            messagebox.showerror("Error", "Testing instruction is empty.")
            return

        run_adb(final_command)

        messagebox.showinfo("Success", "Test completed!")

    send_button = tk.Button(buttons_frame, text="Send Instruction",
                            command=send_instruction,
                            font=("Arial", 16))
    send_button.pack(side='right', padx=10)

    cancel_button = tk.Button(buttons_frame, text="Cancel", command=result_window.destroy, font=("Arial", 16))
    cancel_button.pack(side='right', padx=10)


def get_screen_dimensions():
    try:
        output = subprocess.check_output("adb shell wm size", shell=True)
        output_str = output.decode('utf-8')
        match = re.search(r'Physical size: (\d+)x(\d+)', output_str)
        if match:
            width, height = int(match.group(1)), int(match.group(2))
            return width, height
    except Exception as e:
        print(f"Error getting screen dimensions: {e}")

    return 1080, 1920


# ************
def on_submit():
    messagebox.showinfo("Tips", "Uploaded to chatgpt, processing...")
    d2 = u2.connect()
    d2.uiautomator.start()
    # d2.start_uiautomator()
    text_content = text_box.get("1.0", tk.END)
    open_details_window(text_content)


def on_cancel():
    root.destroy()


root = tk.Tk()
root.title("GPT-Monkey")
root.geometry("610x460")

app_title_label = tk.Label(root, text="GPT-Monkey", font=("Arial", 26))
app_title_label.pack()
d = u2.connect()
d.uiautomator.start()
# d.start_uiautomator()


welcome_label = tk.Label(root, text="Welcome to GPT Monkey.", wraplength=780, justify="left", font=("Arial", 16))
welcome_label.pack(pady=10)

text_box_label = tk.Label(root, text="Please describe the testing requirements", font=("Arial", 16))
text_box_label.pack()

text_box = tk.Text(root, height=10, width=40, font=("Arial", 16))
text_box.pack(pady=10)

text_box.tag_configure("large_font", font=("Arial", 16))

prompt_text = "Enter your testing requirement description here......"
text_box.insert(tk.END, prompt_text, "large_font")

text_box.bind("<FocusIn>", lambda event: text_box.delete("1.0", tk.END) if text_box.get("1.0",
                                                                                        tk.END).strip() == prompt_text else None)

button_frame = tk.Frame(root)
button_frame.pack(side='bottom', anchor='e', pady=10, padx=10)
submit_button = tk.Button(button_frame, text="Confirm", command=on_submit, font=("Arial", 16))
submit_button.pack(side='right', padx=10, pady=10)
cancel_button = tk.Button(button_frame, text="Cancel", command=on_cancel, font=("Arial", 16))
cancel_button.pack(side='right', padx=10, pady=10)

root.mainloop()
