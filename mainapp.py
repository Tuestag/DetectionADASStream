import cv2
import numpy as np
import streamlit as st

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import os
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator


#PROBANDO
import requests
requests.get("https://github.com/Tuestag", headers = {'User-agent': 'your bot 0.1'})

#response = requests.get("https://api.github.com/repos/Tuestag/DetectionCoffeeStream/releases/latest")

class CocoTrainer(DefaultTrainer):

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"

    return COCOEvaluator(dataset_name, cfg, False, output_folder)
  
import base64

########################
main_bg = "BackgroundII.png"
main_bg_ext = "png"

side_bg = "BackgroundII.png"
side_bg_ext = "png"

st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}
   .sidebar .sidebar-content {{
        background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()})
    }}
    </style>
    """,
    unsafe_allow_html=True
)

#######################

@st.cache(persist=True)
def initialization():
    """Loads configuration and model for the prediction.
    
    Returns:
        cfg (detectron2.config.config.CfgNode): Configuration for the model.
        predictor (detectron2.engine.defaults.DefaultPredicto): Model to use.
            by the model.
        
    """
    cfg = get_cfg()
    # Force model to operate within CPU, erase if CUDA compatible devices ara available
    cfg.MODEL.DEVICE = 'cpu'
    # Add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
  #  cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
    # Set threshold for this model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well

    #cfg.MODEL.WEIGHTS = os.path.join("https://api.github.com/repos/Tuestag/DetectionCoffeeStream/releases/assets/49141342")
    ############EDITAR:#################################################
    #cfg.MODEL.WEIGHTS = os.path.join("https://github.com/Tuestag/DetectionCoffeeStream/releases/download/Modelo19CoffeeDetection/modelo19.pth")
    cfg.MODEL.WEIGHTS = os.path.join("https://github.com/Tuestag/DetectionADASStream/releases/download/adasdetectronCO/Detectron2.pth")
    
    from detectron2.data.datasets import register_coco_instances
    register_coco_instances("my_dataset_test", {}, "/test/_annotations.coco.json", "test")
    MetadataCatalog.get("my_dataset_test").thing_classes = ['Adas', 'Moto', 'Obstaculo', 'Peaton', 'Pista', 'Vehiculo']
    MetadataCatalog.get("my_dataset_test").thing_dataset_id_to_contiguous_id = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5:5}
    # Initialize prediction model
    predictor = DefaultPredictor(cfg)

    return cfg, predictor


@st.cache
def inference(predictor, img):
    return predictor(img)


@st.cache
def output_image(cfg, img, outputs):
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get("my_dataset_test"), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    processed_img = out.get_image()

    return processed_img

@st.cache
def onVideo(self,videoPath):
        video = cv2.VideoCapture(videoPath)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        #fourcc = cv2.VideoWriter_fourcc(*'mpv4')
        #out = cv2.VideoWriter("detected_video.mp4", fourcc, 20.0, (w, h))

        #initializing videoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mpv4')
        video_writer = cv2.VideoWriter("output.mp4", fourcc , fps=float(frames_per_second), frameSize=(width, height), isColor=True)

       
        v = VideoVisualizer(MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), ColorMode.IMAGE)

        def runOnVideo(video,maxFrames):# this is for debugging

            readFrames = 0
            while True:
                hasFrame , frame = video.read()
                if not hasFrame:
                    break
                outputs = self.predictor(frame)
                

                frame  = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                visualization = v.draw_instance_predictions(frame, outputs['instances'].to('cpu'))
                visualization = cv2.cvtColor(visualization.get_image(),cv2.COLOR_RGB2BGR)
                yield visualization
                
                readFrames += 1
                if readFrames > maxFrames:
                    break
        num_frames = 200 # here  we reinitialize the number of frames because  of it'll take hours to write the detectedVideos to our video_file and since we don't have a gpu
        # if num_frames is not re-inititialized. the entire frames of the video will be taken into account.. usually taking hours to detect since a 'cpu' and not'gpu' is used
        for visualization in tqdm.tqdm(runOnVideo(video, num_frames), total = num_frames):
            #cv2.imwrite("detected_image.png", visualization)

            video_writer.write(visualization)
        video.release()
        video_writer.release()
        cv2.destroyAllWindows()



@st.cache
def discriminate(outputs, classes_to_detect):
    """Select which classes to detect from an output.
    Get the dictionary associated with the outputs instances and modify
    it according to the given classes to restrict the detection to them
    Args:
        outputs (dict):
            instances (detectron2.structures.instances.Instances): Instance
                element which contains, among others, "pred_boxes", 
                "pred_classes", "scores" and "pred_masks".
        classes_to_detect (list: int): Identifiers of the dataset on which
            the model was trained.
    Returns:
        ouputs (dict): Same dict as before, but modified to match
            the detection classes.
    """
    pred_classes = np.array(outputs['instances'].pred_classes)
    # Take the elements matching *classes_to_detect*
    mask = np.isin(pred_classes, ['Adas', 'Moto', 'Obstaculo', 'Peaton', 'Pista', 'Vehiculo'])
    # Get the indexes
    idx = np.nonzero(mask)

    # Get the current Instance values
    pred_boxes = outputs['instances'].pred_boxes
    pred_classes = outputs['instances'].pred_classes
    pred_masks = outputs['instances'].pred_masks
    scores = outputs['instances'].scores
    st.title(pred_classes)

    # Get them as a dictionary and leave only the desired ones with the indexes
    out_fields = outputs['instances'].get_fields()
    out_fields['pred_boxes'] = pred_boxes[idx]
    out_fields['pred_classes'] = pred_classes[idx]
    out_fields['pred_masks'] = pred_masks[idx]
    out_fields['scores'] = scores[idx]

    return outputs


def main():
    # Initialization
    cfg, predictor = initialization()

    st.title("Detector de Granos de Caf??")
    st.write("Estimado usuario, para hacer uso de esta app debe subir una imagen de granos de caf?? en la cual se realizar?? la detecci??n a partir de la pigmentaci??n que estos presentan. Cada vez que suba una imagen, se reemplazar?? la anterior.")
    
    # Retrieve image
    uploaded_img = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    if uploaded_img is not None:
        file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        # Detection code
        outputs = inference(predictor, img)
        out_image = output_image(cfg, img, outputs)
        st.image(out_image, caption='Processed Image', use_column_width=True)        
        st.write("Muchas gracias por utilizar esta aplicaci??n.")

if __name__ == '__main__':
    main()
