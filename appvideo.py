import streamlit as st
from objectDetection import *
#detector = Detector(model_type='keypointsDetection')

#detector.onVideo("pexels-tima-miroshnichenko-6388396.mp4")
#@st.cache
def func_2(x):
    detector = Detector(model_type=x)
    uploaded_video = st.file_uploader("Upload Video", type = ['mp4','mpeg','mov'])
    if uploaded_video != None:
        
        vid = uploaded_video.name
        with open(vid, mode='wb') as f:
            f.write(uploaded_video.read()) # save video to disk
    
        st_video = open(vid,'rb')
        video_bytes = st_video.read()
        st.video(video_bytes)
        st.write("Uploaded Video")
        detector.onVideo(vid)
        st_video = open('output.mp4','rb')
        video_bytes = st_video.read()
        st.video(video_bytes)
        st.write("Detected Video") 



def main():
    with st.expander("About the App"):
        st.markdown( '<p style="font-size: 30px;"><strong>Welcome to my Object Detection App!</strong></p>', unsafe_allow_html= True)
        st.markdown('<p style = "font-size : 20px; color : white;">This app was built using Streamlit, Detectron2 and OpenCv to demonstrate <strong>Object Detection</strong> in both videos (pre-recorded) and images.</p>', unsafe_allow_html=True)
        


    option = st.selectbox(
     'What Type of File do you want to work with?',
     ('Images', 'Videos'))

    #st.write('You selected:', option)
    if option == "Videos":
        st.title('Object Detection for Videos')
        st.subheader("""
This takes in a video and outputs the video with bounding boxes created around the objects in the video.
""")
        func_2('objectDetection')


if __name__ == '__main__':
		main()
