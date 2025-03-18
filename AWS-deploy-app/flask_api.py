from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
import boto3
from dotenv import load_dotenv
from functions import open_json, smart_video_name, create_embeddings, process_video, model, json_file, collection

load_dotenv()

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(os.getcwd(), os.getenv("VIDEO_UPLOAD_FOLDER"))
filepath = "" # FILEPATH OF VIDEO AFTER ITS UPLOADED
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
s3 = boto3.client('s3',  aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"), aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))
BUCKET_NAME = os.getenv("S3_BUCKET")

@app.route('/upload-video', methods=['POST'])
def upload_video():
    global filepath
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video = request.files['video']
    if video.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Generate a UUID for the video
    video_uuid = str(uuid.uuid4())
    video_file_name = video_uuid + video.filename

    # Determine the file extension
    # filename_parts = os.path.splitext(video.filename)
    # file_extension = filename_parts[1] if len(filename_parts) > 1 else ""  # Handle cases with no extension

    filepath = os.path.join(UPLOAD_FOLDER, video_file_name)
    
    print(type(filepath))
    video.save(filepath)
    try:
        s3.upload_file(filepath, BUCKET_NAME, f"uploads/{video_file_name}")
        print("Video uploaded in S3 bucket")
    except Exception as e:
        print(f"Error uploading video '{video_file_name}' to S3: {e}")
    

    try:
        print(f"file_path = {filepath}")
        process_video(filepath, video_file_name, video_uuid)

        # Open JSON file
        json_file_data = open_json(json_file)

        # Smart Name
        smart_name = smart_video_name(json_file_data)
        
        # Create Embeddings
        create_embeddings(json_file_data, smart_name)

        return jsonify({'status': 'success', 'message': 'Video processing completed', "Smart Name": smart_name})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e), "Here": "Error occured in upload-video API"}), 500

@app.route('/query', methods=['POST'])
def query_data():
    data = request.get_json()
    query_text = data.get('query', '')
    file_uuid = data.get('file_uuid', None)  # Filter results by file_uuid

    if not query_text:
        return jsonify({"error": "Query text is required."}), 400

    if not file_uuid:
        return jsonify({"error": "file_uuid is required to query data from your specific upload."}), 400

    # print("QUERY TEXT = ",query_text)
    try:
        query_embedding = model.encode(query_text)
        # print("Query embeddings done")

        # Query ChromaDB with file_uuid filter
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=1,
            include=["documents", "metadatas"],
            where={"video_uuid": file_uuid}  # Filter results to specific file upload
        )
        # print("RESULTS = ", results)
        response = []
        for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
            response.append({
                **metadata,
                # "filename": metadata['filename'],
                # "timestamp": metadata['timestamp'],
                # "recorded_date": metadata['recorded_date'],
                "result": doc
            })
        # print("RESPONCE = ", response)
        return jsonify(response), 200

    except Exception as e:
        app.logger.error(f"Query error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/videos', methods=['GET'])
def get_videos():
    # video_files = os.listdir(UPLOAD_FOLDER)  # Get uploaded files
    response = s3.list_objects_v2(Bucket=BUCKET_NAME)
    video_files = [obj['Key'] for obj in response.get('Contents', [])]

    video_data = []

    # Fetch video metadata from ChromaDB
    results = collection.get(include=["metadatas"])

    # Organize metadata by video filenames
    video_metadata = {}
    for meta in results["metadatas"]:
        if meta:
            filename = meta["video_file_name"]
            video_uuid = meta["video_uuid"]
            if filename not in video_metadata:
                video_metadata[filename] = video_uuid
            else:
                break

    # Prepare response
    for video in video_files:
        video_data.append({
            "filename": video[44:],
            "video_uuid": video_metadata.get(video[8:], None)  # Map UUID if exists
        })

    return jsonify(video_data)

@app.route("/health", methods=["GET"])
def health_check():
    print("Server up and good")
    return jsonify({"Status": "OK"})
@app.errorhandler(Exception)
def internal_server_error(e):
    print("A 500 error occurred!")  # Log or execute any custom action
    # os.remove(filepath)
    # os.remove(json_file)

    print({"error": "Internal Server Error", "Error Message": e}), 500


if __name__ == '__main__':
    app.run(debug=False,host="0.0.0.0", port=5000)
