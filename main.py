"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import numpy as np
import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

infr_arr = [] #to store the inference time
duration_arr = [] #to store the duration

single_image_mode = False
start_time = 0 
count = 0
total = 0
duration = 0.0


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-s", "--frame_skip_rate", type=float, default=0,
                        help="How much frame to skip"
                        "(0 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client

def draw_boxes(frame, result,prob_threshold, width, height):
    new_conf = 0
    for i,box in enumerate(result[0][0]):
        conf = box[2]
        if conf >= prob_threshold:
            new_conf = conf 
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
    #new_conf will send 0 if confiedence is below prob_threshold else it will send conf
    return frame,new_conf

def postprocess(conf,frame_no,conf_arr,client,fps):
    global start_time,count,total,duration
    if count == 0: 
        if conf > 0:
            count = 1 # if conf is greater than 0 then count will be updated as 1
            total = total + 1 # the total count of the people is increamented by 1
            start_time = frame_no #current frame number is stored in start_frame
    if count == 1:
        post_frames = conf_arr[-3:]
        if np.mean(post_frames) == 0: #if consecutive 3 frame gives 0 then count will be updated as 0
            count = 0
            duration = ((frame_no)-start_time)/fps #duration is calculated by subtracting current frame with start_frame divide by fps.
            client.publish("person/duration", json.dumps({"duration": duration}))
            duration_arr.append(duration)
    client.publish("person", json.dumps({"count": count, "total":total}))
    return

def infer_on_stream(args,client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    
    global single_image_mode
    # Initialise the class
    infer_network = Network()
    
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    plugin = Network()
    plugin.load_model(args.model, args.device, args.cpu_extension)
    net_input_shape = plugin.get_input_shape()
    
    ### TODO: Handle the input stream ###
    if args.input == 'CAM':
        args.input = 0
        
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        single_image_mode = True 
        
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input) 
    
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS) # storing the fps of the video
    
    ### TODO: Loop until stream is over ###
    frame_no = 0
    conf_arr = []
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        frame_no = frame_no+1
        start_infr = time.time()
        if frame_no%(args.frame_skip_rate+1)== 0 or single_image_mode==True:
            # frame will be skipped based on the -s argument (frame_skip_rate) to decrease the inference time.
            if not flag:
                break
            ### TODO: Pre-process the image as needed ###
            key_pressed = cv2.waitKey(60)
            new_frame= np.copy(frame)
            p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
            p_frame = p_frame.transpose((2,0,1))
            p_frame = p_frame.reshape(1, *p_frame.shape)

            ### TODO: Start asynchronous inference for specified request ###
            plugin.exec_net(p_frame)

            ### TODO: Wait for the result ###
            if plugin.wait() == 0:
                ### TODO: Get the results of the inference request ###
                result = plugin.get_output()
                ### TODO: Extract any desired stats from the results ###
                out_frame,conf = draw_boxes(new_frame,result,prob_threshold,width,height)
                conf_arr.append(conf)

                ### TODO: Calculate and send relevant information on ###
                ### current_count, total_count and duration to the MQTT server ###
                ### Topic "person": keys of "count" and "total" ###
                ### Topic "person/duration": key of "duration" ###
                postprocess(conf,frame_no,conf_arr,client,fps)
                infr_arr.append((time.time()-start_infr))

            ### TODO: Send the frame to the FFMPEG server ###
            sys.stdout.buffer.write(out_frame)
            sys.stdout.flush()
            if key_pressed == 27:
                break
            ### TODO: Write an output image if `single_image_mode` ###
            if single_image_mode:
                cv2.imwrite('output_image.jpg', out_frame)
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()
    
def end_result(args):
    #It is used to store the result in result.txt to do better documentation.
    if single_image_mode==True:
        return
    file = open("result.txt", "a")
    data = "Model: {0}\n\
    prob_threshold: {1}\n\
    Total number of people counted: {2}\n\
    Average duration: {3:.2f} sec\n\
    Average inferance time per frame: {4:.2f} sec\n\
    Total inferance time: {5:.2f} min for {6} frames\n\n\
    ".format(args.model,args.prob_threshold,total,np.mean(duration_arr)\
     ,np.mean(infr_arr),np.sum(infr_arr)/60,len(infr_arr))
    file.write(data)
    file.close()

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args,client)
    #upate the result.txt with result
    end_result(args)


if __name__ == '__main__':
    main()
