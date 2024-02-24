import streamlit as st

import warnings
warnings.filterwarnings('ignore')

import cv2
from ultralytics import YOLO
import tempfile

# 1470 cars per 3 busses

AVERAGE_CARBON_EMISSION_RATE_M = 0.19368
# grams of co2 per meter

CAR_MULTIPLIER = 1


def calculate_carbon_emissions(VIDEO_LENGTH_S, SPEED_LIMIT_KMS):

    # convert speed limit to m/s
    SPEED_LIMIT_SPEED_LIMIT_MS = SPEED_LIMIT_KMS / 3.6

    DISTANCE_TRAVELLED_M = VIDEO_LENGTH_S * SPEED_LIMIT_SPEED_LIMIT_MS

    CARBON_EMISSIONS_G_PER_M = DISTANCE_TRAVELLED_M * AVERAGE_CARBON_EMISSION_RATE_M * 1000

    return CARBON_EMISSIONS_G_PER_M


model = YOLO('yolov8n.pt')

title = st.title('C³ - Car Carbon Counter')

st.write('Welcome to C³ - the Car Carbon Counter! This AI-based app, made by João Barbosa, Martim Balthazar, Rafa Leme, and Noah Moesgen for the ISC Wildcode Hackathon, is designed to help engineers, planners, and policymakers understand the carbon emissions of cars on the road. Simply upload a video and we will do the rest! Images are treated as 30 second long videos.')

def frame_splitter(video_cap):
    # get first frame of video
    ret, frame = video_cap.read()

    # save first frame to temp file
    cv2.imwrite('temp.jpg', frame)

def predict():
    image_path = 'temp.jpg'

    results = model([
        "temp.jpg"
    ])

    results = model.predict(source=image_path,
                            imgsz=640,  # Resize image to 640x640 (the size pf images the model was trained on)
                            conf=0.1,
                            )   # Confidence threshold: 50% (only detections above 50% confidence will be considered)

    return results

selected = None

with st.sidebar:
    # selectable

    #st.radio('Footage', ['1', '2', '3'])

    st.title('Sample Data')

    speed_limit = st.number_input('Insert the Speed Limit (km/h)', step=5, value=40)

    selected = st.selectbox('Footage', ['Video 1', 'Video 2', 'Video 3'])

    with st.expander('View Sample Footage'):
        st.image('1.jpg', caption="Video 1 (30s)", use_column_width=True)
        st.image('2.jpg', caption="Video 2 (30s)", use_column_width=True)
        st.image('3.jpg', caption="Video 3 (30s)", use_column_width=True)

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
                    if box.cls in match_types:
                        CAR_COUNT += 1
                
                sample_image = results.plot(line_width=2)

                # from cv2 docs
                sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
                st.image(sample_image)

                video_cap = cv2.VideoCapture(image_path)

                fps = video_cap.get(cv2.CAP_PROP_FPS)
                totalNoFrames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
                                    
                VIDEO_LENGTH = 30

                print(VIDEO_LENGTH)

                CARBON_EMISSIONS_G_PER_M = calculate_carbon_emissions(VIDEO_LENGTH, speed_limit)

                st.text(f"Car Count: {CAR_COUNT}")

                CARBON_EMISSIONS_FINAL = CARBON_EMISSIONS_G_PER_M * CAR_COUNT * CAR_MULTIPLIER

                st.text(f"Carbon Emissions Detected: {format_grams(round(CARBON_EMISSIONS_FINAL, 2))} of CO2 per kilometer of road travelled")

    st.button('Submit', on_click=predict_sample)

def format_grams(grams):
    if grams < 1000:
        return f"{round(grams, 2)} grams"
    else:
        return f"{round(grams / 1000, 2)} kilograms"

with st.form("my_form"):
    speed_limit = st.number_input('Insert the Speed Limit (km/h)', step=5, value=40)
    #video_bool = st.toggle('Video Mode', value=False)

    file_uploader = st.file_uploader('Upload Footage Here', accept_multiple_files=False, type=['mp4', 'avi', 'mov', 'flv', 'wmv', 'mkv', 'webm', 'png', 'jpg', 'jpeg'])

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")

    if submitted:
        if file_uploader is None:
            st.error('Please upload a file')
            st.stop()

        # https://blog.jcharistech.com/2021/01/21/how-to-save-uploaded-files-to-directory-in-streamlit-apps/
        if file_uploader is not None:
            #file_details = {"FileName": file_uploader.name, "FileType": file_uploader.type}

            file_uploaded = tempfile.NamedTemporaryFile(delete=False)

            file_uploaded.write(file_uploader.read())

            video_cap = cv2.VideoCapture(file_uploaded.name)

            fps = video_cap.get(cv2.CAP_PROP_FPS)
            totalNoFrames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)

            # if photo, make the video 1 second
            print(file_uploader.type)
            if file_uploader.type in ['image/png', 'image/jpg', 'image/jpeg']:
                VIDEO_LENGTH = 30
            else:
                st.text("FILE IS A VIDEO")
                VIDEO_LENGTH = totalNoFrames / fps
        
            # OpenCV VideoCapture using the uploaded file name
            #video_file = cv2.VideoCapture(f'{file_uploader.name}.{file_uploader.type}')

            # https://stackoverflow.com/questions/49048111/how-to-get-the-duration-of-video-using-opencv
            #VIDEO_LENGTH = totalNoFrames / fps

            # get frame
            if file_uploader.type not in ['image/png', 'image/jpg', 'image/jpeg']:
                frame_splitter(video_cap)
            else:
                #cv2.imwrite('temp.jpg', file_uploaded.name)
                # create file for image
                #with open('temp.jpg', 'wb') as f:
                #    f.write(file_uploaded.read())
                cv2.imwrite('temp.jpg', cv2.imread(file_uploaded.name))
                

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
                if box.cls in match_types:
                    CAR_COUNT += 1
            
            sample_image = results.plot(line_width=2)

            sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
            st.image(sample_image)

            CARBON_EMISSIONS_G_PER_M = calculate_carbon_emissions(VIDEO_LENGTH, speed_limit)

            st.text(f"Video is {round(VIDEO_LENGTH)} seconds long; Car Count: {CAR_COUNT}")

            CARBON_EMISSIONS_FINAL = CARBON_EMISSIONS_G_PER_M * CAR_COUNT * CAR_MULTIPLIER

            st.text(f"Carbon Emissions Detected: {format_grams(round(CARBON_EMISSIONS_FINAL, 2))} of CO₂ per kilometer of road travelled")

