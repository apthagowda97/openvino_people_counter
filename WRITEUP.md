# Project Write-Up

I used [faster_rcnn_inception_v2_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz) from tensorflow object detection api and I got the expected result.

## Explaining Custom Layers

Custom layers are layers that are not included in a list of known layers. If my model contains any layers that are not in the list of known layers, the Model Optimizer classifies them as custom and it has to be handled separately.

As the faster_rcnn_inception_v2_coco belongs to the frozen topologies from TensorFlow Object Detection Models Zoo and there are no custom layers in it. The model is converted to IR representation by using model optimizer without custom layers as all the layers are in the known list of layers.

## Comparing Model Performance

The below table will compare the models before and after conversion to Intermediate Representation along with intel pre-trained model.

#### [NOTE: The intel pretrained model(person-detection-retail-0013) is used just to comapare as an extra, my choosen model is faster_rcnn_inception_v2_coco]

| Models| Total people count|Average Duration| inference time /frame | Totoal inference time | Frames skip rate | Threshold | Precision | Size |
| ------------- |:-------------:| :-----:| :-----: | :----: |  :-----: | :----: | :----: | :----: |
|person-detection-retail-0013| 6 | 17.92 sec | 0.12 sec |  0.55 min | 4 | 0.6 | fp16 | 1.52mb |
|faster_rcnn_inception_v2_coco(after conversion to IR)| 6 | 18.58 sec | 1.04 sec | 4.81 min | 4 | 0.8 | fp16 | 25.5mb* |
|faster_rcnn_inception_v2_coco(before conversion to IR)| 6 | 18.67 sec | 1.39 sec | 6.43 min | 4 | 0.8 | fp32 | 166mb |

**Only frozen_inference_graph.xml and frozen_inference_graph.bin are taken into account*


- The faster rcnn model is huge and it takes a lot of inference time. So to make it faster without compromising the results I developed the method of skipping some frames between two frames which are fed into the model.
(To make it easier to judge I applied the method of skipping frames even for Intel pre-trained model even though it takes less inference time)

- Now talking about the model performance it can be seen that all 3 models perform same in terms of total people count or average duration but when it comes to inference time the Intel pre-trained model is the winner.

- Comparing the pre and post IR of faster_rcnn_inception_v2_coco models the IR decreases the total inference time by 2 minutes from 6.43 min to 4.81 min without compromising the accuracy of the model and also decreases the size of the model by 144mb to 25.5mb.

- The inference time and the size of the model is the key when deploying pipeline in edge devices and openvino helps to achieve this without compromising the result.

## Assess Model Use Cases

- ATMs:
  - It can be deployed in the edge camera to monitor the movement of a person in ATM and can raise alarm when some person stays way longer than average time in front of the ATM.

- Banks/Stores/Malls:
  - It can help in sending alerts to the nearby police station during the burglary.
 
- Social distancing:
  - Going by the current situation of COVID 19 pandemic. The camera feed can be used to maintain or limit the number of person inside the store/mall and thereby maintaining social distancing.

  - By deploying it in the entry and exit door cameras it can get the total number of people inside the store by subtracting the total count from the entrance with the exit and can alert the security/authority that a maximum number of people are inside the store, which will allow them to take further steps.


## Assess Effects on End-User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

- Lighting: The model may not work in poor lighting condition or in the night time. Even if the camera can capture in the night it still gives less accuracy.

- Accuracy: Model accuracy may be affected if objects like boxes or furniture which are of the size of a person are detected as a person. It is not suitable for very critical application but can be managed by re-training the model using specific data and implementing the pipeline using the newly trained model.

- Focal length/Image size: If the camera is not properly focusing on the subjected area it may not detect the person or it may lead to false detection of the person. Similarly, if the size of the image is low/less resolution the accuracy of the model decreases.
 
## Model Research

### faster_rcnn_inception_v2_coco:

I was able to achieve the correct prediction in terms of people count, total and average duration by using TensorFlow object detection API model faster_rcnn_inception_v2_coco.

Link to the [faster_rcnn_inception_v2_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz ) model.
#### Running the faster_rcnn_inception_v2_coco model on the input video using google colab(before conversion to IR):

I have written the colab notebook to run the  faster_rcnn_inception_v2_coco model on the input video.

- [notebook](notebook/people_counter_faster_rcnn_colab/people_counter_faster_rcnn.ipynb)

- [output video](notebook/people_counter_faster_rcnn_colab/out.mp4)

#### Steps to convert faster_rcnn_inception_v2_coco to IR using model optimizer

Change the directory to models/download

```
cd /home/workspace/models/download
```
Download the model using wget.

```
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz 
```
Unzip the model.
```
tar -xvf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz 
```
Go inside the directory.
```
cd  /home/workspace/models/download/faster_rcnn_inception_v2_coco_2018_01_28
```
Use the model optimizer to convert the faster_rcnn_inception_v2_coco using frozen_inference_graph.pb and pipeline.config and give they data_type as FP16.
Store the result .xml and .bin files in the faster_rcnn_inception_v2_coco directory.

```
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --data_type FP16 --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json  -o /home/workspace/models/faster_rcnn_inception_v2_coco
```
After converting the model there is no need of the original model and it can be deleted by deleting the download folder.

## Run the application

From the main directory:

### Step 1 - Start the Mosca server

```
cd webservice/server/node-server
node ./server.js
```

You should see the following message, if successful:
```
Mosca server started.
```

### Step 2 - Start the GUI

Open a new terminal and run below commands.
```
cd webservice/ui
npm run dev
```

You should see the following message in the terminal.
```
webpack: Compiled successfully
```

### Step 3 - FFmpeg Server

Open a new terminal and run the below commands.
```
sudo ffserver -f ./ffmpeg/server.conf
```

### Step 4 - Run the code

Open a new terminal to run the code. 

#### Setup the environment

You must configure the environment to use the Intel® Distribution of OpenVINO™ toolkit one time per session by running the following command:
```
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
```

You should also be able to run the application with Python 3.6, although newer versions of Python will not work with the app.

#### Running on the CPU

When running Intel® Distribution of OpenVINO™ toolkit Python applications on the CPU, the CPU extension library is required. This can be found at: 

```
/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/
```

*Depending on whether you are using Linux or Mac, the filename will be either `libcpu_extension_sse4.so` or `libcpu_extension.dylib`, respectively.* (The Linux filename may be different if you are using an AVX architecture)

*Though by default application runs on CPU, this can also be explicitly specified by ```-d CPU``` command-line argument*

*The argument ```-s ``` called frame_skip_rate which will skip the number of frames given before sending next frame to inference every time.*

*The argument ```-pt``` called probability threshold which will help in discarding the frames below that threshold.*

#### To run the faster_rcnn_inception_v2_coco(single image mode)

```
python main.py -i resources/single_image.jpg -m models/faster_rcnn_inception_v2_coco/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 -s 4
```
It will create the output file [output_image.jpg](/output_image.jpg)

#### To run the faster_rcnn_inception_v2_coco

```
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/faster_rcnn_inception_v2_coco/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.8 -s 4 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```

#### Note: Every time any model is run the results will be updated in [result.txt](result.txt).

If you are in the classroom workspace, use the “Open App” button to view the output. If working locally, to see the output on a web-based interface, open the link [http://0.0.0.0:3004](http://0.0.0.0:3004/) in a browser.