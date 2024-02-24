import streamlit as st

# Disable warnings in the notebook to maintain clean output cells
import warnings
warnings.filterwarnings('ignore')

import cv2
from ultralytics import YOLO
import tempfile

#col1, col2 = st.columns(2)

# 1470 cars per 3 busses

AVERAGE_CARBON_EMISSION_RATE_M = 0.19368
# grams of co2 per meter

CAR_MULTIPLIER = 1.1


def calculate_carbon_emissions(VIDEO_LENGTH_S, SPEED_LIMIT_KMS):

    # convert speed limit to m/s
    SPEED_LIMIT_SPEED_LIMIT_MS = SPEED_LIMIT_KMS / 3.6

    DISTANCE_TRAVELLED_M = VIDEO_LENGTH_S * SPEED_LIMIT_SPEED_LIMIT_MS

    CARBON_EMISSIONS_G_PER_M = DISTANCE_TRAVELLED_M * AVERAGE_CARBON_EMISSION_RATE_M * 1000

    return CARBON_EMISSIONS_G_PER_M


model = YOLO('yolov8n.pt')

title = st.title('C³ - Car Carbon Counter')

st.write('Welcome to C³ - the Car Carbon Counter! This AI-based app, made by João Barbosa, Martim Balthazar, Rafa Leme, and Noah Moesgen for the ISC Wildcode Hackathon, is designed to help engineers, planners, and policymakers understand the carbon emissions of cars on the road. Simply upload a video and we will do the rest!')

def frame_splitter(video_cap):
    # get first frame of video
    ret, frame = video_cap.read()

    # save first frame to temp file
    cv2.imwrite('temp.jpg', frame)

def predict():
    # Path to the image file
    image_path = 'temp.jpg'

    results = model([
        "temp.jpg"
    ])

    # Perform inference on the provided image(s)
    results = model.predict(source=image_path,
                            imgsz=640,  # Resize image to 640x640 (the size pf images the model was trained on)
                            conf=0.1,
                            )   # Confidence threshold: 50% (only detections above 50% confidence will be considered)

    return results

selected = None

with st.sidebar:
    # selectable

    # make the images actually clickable
    # 3 images
    # 1.jpg, 2, 3

    #st.radio('Footage', ['1', '2', '3'])

    st.title('Sample Data')

    speed_limit = st.number_input('Insert the Speed Limit (km/h)', step=5, value=40)

    selected = st.selectbox('Footage', ['Video 1', 'Video 2', 'Video 3'])

    with st.expander('View Sample Footage'):
        st.image('1.jpg', caption="Video 1", use_column_width=True)
        st.image('2.jpg', caption="Video 2", use_column_width=True)
        st.image('3.jpg', caption="Video 3", use_column_width=True)

        def predict_sample():
            with st.expander('Results - Sample Footage'):
                image_path = f'{selected.split(" ")[1]}.jpg'

                results = model.predict(source=image_path,
                                        imgsz=640,  # Resize image to 640x640 (the size pf images the model was trained on)
                                        conf=0.1,   # Confidence threshold: 50% (only detections above 50% confidence will be considered)
                                        classes=[2, 3, 5, 7]
                                        )
                

                results = results[0]

                match_types = [2, 3, 5, 7]
                CAR_COUNT = 0

                for box in results.boxes:
                    print(box.cls, box.conf)
                    if box.cls in match_types:
                        CAR_COUNT += 1
                
                sample_image = results.plot(line_width=2)

        # Convert the color of the image from BGR to RGB for correct color representation in matplotlib
                sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
                st.image(sample_image)

                video_cap = cv2.VideoCapture(image_path)

                fps = video_cap.get(cv2.CAP_PROP_FPS)
                totalNoFrames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
                                    
                VIDEO_LENGTH = totalNoFrames / fps

                CARBON_EMISSIONS_G_PER_M = calculate_carbon_emissions(VIDEO_LENGTH, speed_limit)

                st.text(f"Car Count: {CAR_COUNT}")

                CARBON_EMISSIONS_FINAL = CARBON_EMISSIONS_G_PER_M * CAR_COUNT * CAR_MULTIPLIER

                st.text(f"Carbon Emissions Detected: {round(CARBON_EMISSIONS_FINAL, 2)} grams of CO2 per kilometer of road travelled")

    st.button('Submit', on_click=predict_sample)



# 2 col wide form like so 3 in total
    # make 3 selectable options for the user to choose from of images

with st.form("my_form"):
    speed_limit = st.number_input('Insert the Speed Limit (km/h)', step=5, value=40)
    #video_bool = st.toggle('Video Mode', value=False)

    file_uploader = st.file_uploader('Upload Footage Here', accept_multiple_files=False, type=['mp4', 'avi', 'mov'])

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")

    if submitted:
        if file_uploader is None:
            st.error('Please upload a video file')
            st.stop()

        # https://blog.jcharistech.com/2021/01/21/how-to-save-uploaded-files-to-directory-in-streamlit-apps/
        if file_uploader is not None:
            #file_details = {"FileName": file_uploader.name, "FileType": file_uploader.type}

            file_uploaded = tempfile.NamedTemporaryFile(delete=False)

            file_uploaded.write(file_uploader.read())

            video_cap = cv2.VideoCapture(file_uploaded.name)

            fps = video_cap.get(cv2.CAP_PROP_FPS)
            totalNoFrames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
            # OpenCV VideoCapture using the uploaded file name
            #video_file = cv2.VideoCapture(f'{file_uploader.name}.{file_uploader.type}')

            # https://stackoverflow.com/questions/49048111/how-to-get-the-duration-of-video-using-opencv

            # get frame
            frame_splitter(video_cap)

            image_path = 'temp.jpg'

            results = model.predict(source=image_path,
                                    imgsz=640,  # Resize image to 640x640 (the size pf images the model was trained on)
                                    conf=0.1,   # Confidence threshold: 50% (only detections above 50% confidence will be considered)
                                    classes=[2, 3, 5, 7]
                                    )
            
            #print(results)

            results = results[0]

            match_types = [2, 3, 5, 7]
            CAR_COUNT = 0

            for box in results.boxes:
                print(box.cls, box.conf)
                if box.cls in match_types:
                    CAR_COUNT += 1
            
            sample_image = results.plot(line_width=2)

# Convert the color of the image from BGR to RGB for correct color representation in matplotlib
            sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
            st.image(sample_image)
                                
            VIDEO_LENGTH = totalNoFrames / fps

            CARBON_EMISSIONS_G_PER_M = calculate_carbon_emissions(VIDEO_LENGTH, speed_limit)

            st.text(f"Video is {round(VIDEO_LENGTH)} seconds long; Car Count: {CAR_COUNT}")

            CARBON_EMISSIONS_FINAL = CARBON_EMISSIONS_G_PER_M * CAR_COUNT * CAR_MULTIPLIER

            st.text(f"Carbon Emissions Detected: {round(CARBON_EMISSIONS_FINAL, 2)} grams of CO2 per kilometer of road travelled")
