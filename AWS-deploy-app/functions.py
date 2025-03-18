import os
# os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin")
# os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\include")
# os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\lib")
# os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\lib\x64")
# os.add_dll_directory(r"C:\Users\Admin\Desktop\opencv-installation\binaries\install\x64\vc17\bin")
import cv2
from ultralytics import YOLO
import base64
from groq import Groq
import json
from datetime import datetime
from together import Together
import uuid
import shutil
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from flask import jsonify
# images_folder = "Extracted-SS"

load_dotenv()

extracted_frame_folder = os.path.join(os.getcwd(), os.getenv("FRAMES_EXTRACT_FOLDER"))
json_file = os.path.join(os.getcwd(),os.getenv("JSON_FILE_PATH"))  # Replace with your desired output folder path
frame_interval_seconds = 13

model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True)
model.max_seq_length = 8192

client = chromadb.HttpClient(host=os.getenv("CHROMADB_HOST"), port=8000)
collection = client.get_or_create_collection(name="Embeddings")



def play_and_extract_frames(video_path, output_folder, frame_interval_seconds):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Open the video file
    print(f"video_path inside extract frames = {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Get frame rate of the video
    interval_frames = int(frame_rate * frame_interval_seconds)  # Frames to skip based on interval
    # delay = int(1000 / frame_rate)  # Delay between frames in milliseconds for real-time playback
    
    frame_count = 0
    frame_number = 0
    
    while True:
        # start_time = time.time()  # Record start time for precise delay handling
        
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break  # Exit the loop if no more frames are available
        
        # Display the video frame
        # cv2.imshow("Video", frame)
        
        # Save frames at the specified interval
        if frame_count % interval_frames == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            frame_filename = f"Frame{frame_number}_{timestamp}.jpg"
            frame_path = os.path.join(output_folder, frame_filename)
            cv2.imwrite(frame_path, frame)
            print(f"Saved: {frame_path}")
            frame_number += 1
        
        frame_count += 1
        
        # # Wait for the calculated delay or until 'q' is pressed
        # elapsed_time = (time.time() - start_time) * 1000  # Elapsed time in milliseconds
        # remaining_delay = max(1, delay - int(elapsed_time))  # Ensure a minimum delay of 1ms
        # if cv2.waitKey(remaining_delay) & 0xFF == ord('q'):
        #     print("Playback interrupted by user.")
        #     break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Playback and frame extraction complete.")


def yolo_11x(image):
    model = YOLO("yolo11x-pose.pt")
    result = model(image)
    for r in result:
        summary = r.summary()
    return summary


def llama_vision(image_path):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    client = Groq(api_key="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")

    # Chat completion request
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "Analyze the given image and provide the following details:\n"
                        "1. Count of people present in the image.\n"
                        "2. Identify activities happening in the image (e.g., criminal avtivity, detect weapons, working on a computer, having a conversation, idle, etc.).\n"
                        "3. Determine whether individuals are is alone, in groups, or not engaged in work, or are walking. Identify there pose\n"
                        "4. **Important**: Provide detailed observations about the environment, and identify interaction between various people and objects.\n"
                        "5. Highlight any unusual or suspicious behavior (if any).\n"
                        "\n"
                        "Format the response as a structured report for easy understanding."
                    )},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        model="llama-3.2-90b-vision-preview",
        temperature=1,  # Slightly reduced for more deterministic results
        top_p=1,        # Focus on high-probability completions
        stream=False,
        stop=None,
        max_completion_tokens=512,
    )

    # Print the response
    llama_vision_responce = chat_completion.choices[0].message.content

    return llama_vision_responce


model_prompt = """
You will analyze data from two sources to provide a final verdict on the scene:  
1. **Llama-Vision Data**: High-level descriptive overview.  
2. **YOLO11x-Pose Detection Data**: Detailed pose and bounding box data for individuals in the scene. \n
"I will give you context of your previous responce (in case of first iteration it might be empty). In available you should strictly mention the changes that have happened from your previous responce **If none end the responce**." 

### Your Tasks:  
1. Compare both datasets and resolve discrepancies, highlighting key findings.  
2. For each person detected, determine:  
   - Activity (sitting, standing, walking).  
   - Proximity to others (in a crowd or not).  
3. Provide a **final integrated verdict** of the scene:  
   - Number of people, their activities, interactions, and any notable details.  
   - Environmental context and any unusual behaviors.  
   - Summary of key insights prioritized by importance. 
#  \n\n#### Comparison and Discrepancy Resolution ** and strictly focus on giving final verdict
  """
def llama_3_3(yolo_data,llama_vision_data, previous_llama3_3_responce):

    client = Together(api_key="7c9e101c0a6227a7f141531b80935867ffb69cf6e7dc5b976eeef779cc0c4cb3")

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=[
            {
                "role": "user",
                "content": f"this is your previous responce{previous_llama3_3_responce}"
                """You will analyze data from two sources to provide a final verdict on the scene:  
                1. **Llama-Vision Data**: High-level descriptive overview.  
                2. **YOLO11x-Pose Detection Data**: Detailed pose and bounding box data for individuals in the scene. \n
                "I will give you context of your previous responce (in case of first iteration it might be empty). In available you should strictly mention the changes that have happened from your previous responce **If none end the responce**." 

                ### Your Tasks:  
                1. Compare both datasets and resolve discrepancies, highlighting key findings.  
                2. For each person detected, determine:  
                - Activity (sitting, standing, walking).  
                - Proximity to others (in a crowd or not).  
                3. Provide a **final integrated verdict** of the scene:  
                - Number of people, their activities, interactions, and any notable details.  
                - Environmental context and any unusual behaviors.  
                - Summary of key insights prioritized by importance. 
                #  \n\n#### Comparison and Discrepancy Resolution ** and strictly focus on giving final verdict"""
            f"\n\nHere is the data for your analysis:\n\n"
            f"### llama-vision data:\n{llama_vision_data}\n\n"
            f"### yolo11x-pose detection data:\n{yolo_data}\n\n"
            "Now, combine these datasets and provide the best possible output."
            "## at last summarise the output based on the importance you think is best"
        }
        ],
        max_tokens=512,
        temperature=0.7,
        top_p=0.7,
        top_k=50,
        repetition_penalty=1,
        stop=["<|eot_id|>","<|eom_id|>"],
        stream=False
    )
    
    return response.choices[0].message.content




    client = Groq(api_key="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")

    # Chat completion request
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": (
            #         "You will analyze and combine data from two sources: \n"
            # "1. **llama-vision data**: Provides a high-level descriptive analysis of a scene.\n"
            # "2. **yolo11x-pose detection data**: Offers detailed pose and bounding box information for individuals in the scene.\n\n"
            # "Your task:\n"
            # "1. Integrate the information from both sources to create a comprehensive understanding of the scene.\n"
            # "2. Resolve any differences between the datasets, such as the number of people detected, and highlight key insights.\n"
            # "3. For each person detected, identify:\n"
            # "   - Whether they are sitting, standing, walking, or in a crowd.\n"
            # "     - Use the **bounding box ('box')** information to detect if a person is in a crowd (e.g., overlapping or closely packed boxes).\n"
            # "     - Use the **pose keypoints ('keypoints')** data to infer their posture and movement (e.g., position of shoulders, elbows, knees).\n"
            # "4. Provide a detailed summary of the scene, including:\n"
            # "   - The number of people present.\n"
            # "   - Activities of each person.\n"
            # "   - Environmental context (e.g., lighting, setting).\n"
            # "   - Any unusual or suspicious behaviors.\n"
            # "5. Highlight your approach for identifying sitting, standing, walking, and crowded individuals, and explain any assumptions made.\n\n"
            f"this is your previous responce{previous_llama3_3_responce}"
            f"{model_prompt}\n\nHere is the data for your analysis:\n\n"
            f"### llama-vision data:\n{llama_vision_data}\n\n"
            f"### yolo11x-pose detection data:\n{yolo_data}\n\n"
            "Now, combine these datasets and provide the best possible output."
            "## at last summarise the output based on the importance you think is best"
                    )},
                ],
            }
        ],
        model="llama-3.2-3b-preview",
        temperature=0.8,  # Slightly reduced for more deterministic results
        top_p=1,        # Focus on high-probability completions
        stream=False,
        stop=None,
        max_completion_tokens=128,
    )
    return chat_completion.choices[0].message.content

def open_json(filepath):
    # encoding = detect_encoding(filepath)
    with open (filepath, "r") as data:
        return json.load(data)

def delete_files_from_folder(folder_path):
    try:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            os.remove(file_path)
        print("Files removed")
    except FileNotFoundError:
        print(f"Error: Folder '{folder_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def write_in_json(frame_name,llama_3_3_result, video_filename, video_uuid, json_file):
    
    os.makedirs(os.path.dirname(json_file), exist_ok=True)

    # Check if JSON file exists and is valid
    if not os.path.exists(json_file) or os.path.getsize(json_file) == 0:
        with open(json_file, "w") as f:
            json.dump([], f)
    # Open and update the JSON file
    with open(json_file, "r+") as f:
        try:
            data = json.load(f)  # Load existing data
        except json.JSONDecodeError:
            data = []  # Initialize to an empty list if the file is invalid

        # Append the new result
        data.append({
            "video_uuid":video_uuid,
            "video_file": video_filename,
            "frame_name": frame_name,  # Replace with the actual filename
            "llama_3_3_result": llama_3_3_result  # Replace with the actual result
        })

        # Write the updated data back to the file
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()  # Ensure no leftover content remains


def smart_video_name(data):
    llama_res = ""
    split_len = 0

    for item in data:
        res = item.get("llama_3_3_result")
        split_len += len(res.split())
        if split_len < 4500:
            llama_res = llama_res + res + "\n\n\n"
        else:
            break

    client = Together(api_key="7c9e101c0a6227a7f141531b80935867ffb69cf6e7dc5b976eeef779cc0c4cb3")

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[
            {
                "role": "user",
                "content": "I need one smart video title about the content inside video\nno extra information:\n"
                            f"This is data {llama_res}"
            }   
        ],
        max_tokens=50,
        temperature=1,
        top_p=0.5,
        top_k=50,
        repetition_penalty=4,
        stop=["<|eot_id|>","<|eom_id|>"],
        stream=False
    )
    return response.choices[0].message.content

def final_output():
    client = Groq(api_key="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")

    with open(r"P:\Communication Crafts\groq-llama-vision\Project files\json-file\result-3.json", "r") as data:
         json_data = json.load(data)
    
    
    result_parts = []
    for entry in json_data:
        if "filename" in entry and "llama_3_3_result" in entry:
            result_parts.append(f"{entry['filename']}\n{entry['llama_3_3_result']}")

    # Combine all into a single string
    result_string = "\n\n".join(result_parts)

    # Chat completion request
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                            "You will get the data generated by llama3.3 model."
                            "This is the data of the images/frames that was initially extracted from video."
                            "You are instructed to combine all the information and generate a legitimate and consise response and give final report by visualizing what might have happned in the video from the given combined data"
                            f"Here is the data:\n{result_string}"
                    )},
                ],
            }
        ],
        model="llama-3.2-3b-preview",
        temperature=1,  # Slightly reduced for more deterministic results
        top_p=1,        # Focus on high-probability completions
        stream=False,
        stop=None,
        max_completion_tokens=512,
    )
    return chat_completion.choices[0].message.content


# if __name__ == "__main__":
    
#     video_path = r"P:\Communication Crafts\groq-llama-vision\Project files\test-videos\istockphoto-1729752794-640_adpp_is.mp4"  # Replace with your video file path
#     output_folder = r"P:\Communication Crafts\groq-llama-vision\Project files\Extracted-SS"  # Replace with your desired output folder path
#     frame_interval_seconds = 1  # Change this to the interval in seconds (e.g., 1, 2, 5)
#     json_file = r"P:\Communication Crafts\groq-llama-vision\Project files\json-file\result-3.json"

#     play_and_extract_frames(video_path, output_folder, frame_interval_seconds)

#     previous_llama3_3_responce=""
#     for filename in os.listdir(images_folder):
#         file_path = os.path.join(images_folder, filename)

#         # Check if the file is an image (you can refine this check as needed)
#         if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
#             print(f"Processing image: {filename}")

            
#             llama_vision_responce = llama_vision(file_path)
#             yolo_response = yolo_11x(file_path)
#             llama_3_3_result = llama_3_3(yolo_response, llama_vision_responce, previous_llama3_3_responce)
#             previous_llama3_3_responce = llama_3_3_result



#             write_in_json(filename, llama_3_3_result, json_file)
#             # print(f"\n\n\nLlama Vision responce:\n{llama_vision_responce}")
#             # print(f"Llama 3.3 Response: {llama_3_3_result}")

    
#     end_result = final_output()
#     print(end_result)

    

def create_embeddings(data, smart_name):
    try:
        global model, collection, json_file
        for item in data:
            video_uuid = item.get("video_uuid")
            video_file_name = item.get("video_file")
            frame_name = item.get("frame_name")
            image_results = item.get("llama_3_3_result")
            filename_plus_uuid = f"{video_file_name}_{video_uuid}"

            embeded_image_result = model.encode(image_results)

            collection.add(
                ids= [f"{video_uuid}_{frame_name}"],
                embeddings=[embeded_image_result],
                documents=[image_results],
                metadatas=[{
                    "video_uuid": video_uuid,
                    "video_file_name" : video_file_name,
                    "frame_name": frame_name,
                    "filename_plus_uuid": filename_plus_uuid,
                    "smart_name": smart_name
                }]
            )
            # delete_files_from_folder(json_file)
        os.remove(json_file)
    except Exception as e:
        print("Errors while creating embeddings:", e)
        return jsonify({"Errors while creating embeddings" : e})


def process_video(video_path, video_filename, video_uuid):
    global extracted_frame_folder, frame_interval_seconds
    previous_llama3_3_responce=""

    unique_folder_name = str(uuid.uuid4())
    print(unique_folder_name)
    output_folder = os.path.join(extracted_frame_folder, unique_folder_name)
    print(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # Replace this with your actual processing logic
    # print(f"Processing video: {video_path}")
    # Simulate processing
    try:
        play_and_extract_frames(video_path, output_folder, frame_interval_seconds)

        for frame_name in os.listdir(output_folder):
            file_path = os.path.join(output_folder, frame_name)

            # Check if the file is an image (you can refine this check as needed)
            if frame_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                # print(f"Processing image: {frame_name}")
                
                llama_vision_responce = llama_vision(file_path)
                # print("\n\nLlama Vision Answer:\n",llama_vision_responce)
                yolo_response = yolo_11x(file_path)
                llama_3_3_result = llama_3_3(yolo_response, llama_vision_responce, previous_llama3_3_responce)
                previous_llama3_3_responce = llama_3_3_result

                write_in_json(frame_name, llama_3_3_result, video_filename, video_uuid, json_file)
    except Exception as e:
        print(f"LOGS = {e}")
    finally:
        # delete_files_from_folder(extracted_frame_folder)
        shutil.rmtree(output_folder)
        print(f"Folder {output_folder} deleted.")
        os.remove(video_path)
        print("Video Deleted from local storage")


    # subprocess.run(['echo', f'Processing {frame_name}'])
# def detect_encoding(filepath):
#     with open(filepath, "rb") as f:
#         raw_data = f.read()
#         result = chardet.detect(raw_data)
#         return result["encoding"]
