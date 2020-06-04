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
import time

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

from yolo import YoloParams, parse_yolo_region

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


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
    parser.add_argument("-it", "--IoU_threshold", type=float, default=0.5,
                        help="Intersection over Union threshold for detections same object"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client

def draw_boxes(frame, result, args, width, height):
    '''
    Draw bounding boxes onto the frame.
    '''
    for box in result[0][0]: 
        if box[1]!=1:
            continue
        conf = box[2]
        if conf >= args.prob_threshold:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255,0,0), 1)
    return frame

def draw_result_boxes(frame, result, width, height):
    '''
    Draw bounding boxes onto the frame.
    '''
    for r in result: 
        xmin = int(r['box'][0] * width)
        ymin = int(r['box'][1] * height)
        xmax = int(r['box'][2] * width)
        ymax = int(r['box'][3] * height)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255,0,0), 1)
    return frame

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def process_ssd_result(result, args, cnt_frame, fps, result_info):
    newObj = 0
    new_result_info = []
    last_visit = None

    boxes = result[0][0]

    # reduce overlap boxes
    boxes = sorted(boxes, key=lambda obj : (obj[1],obj[2]), reverse=True)
    for i in range(len(boxes)):
        if boxes[i][2] == 0:
            continue
        for j in range(i + 1, len(boxes)):
            if boxes[i][1] != boxes[j][1]:
                continue
            iou = bb_intersection_over_union(boxes[i][3:], boxes[j][3:])
            #print('iou', i, j, iou)
            if iou > args.IoU_threshold:
                boxes[j][2] = 0
    # filter label = person and probaibity > args.pt
    boxes = list(filter(lambda obj: obj[1] == 1 and args.prob_threshold < obj[2], boxes))
    
    # process boxes
    for box in boxes:
        newbox = box[3:]
        #print(newbox)
        if len(result_info) == 0:
            new_result_info.append({'start':cnt_frame,'end':cnt_frame,'box':newbox,'new':1})
        else:
            found = False
            for i in range(len(result_info)):
                obj = result_info[i]
                if obj['end']==cnt_frame:
                    break
                iou = bb_intersection_over_union(obj['box'], newbox)
                #print('iou:',obj['box'], newbox, iou)
                if iou > args.prob_threshold:
                    obj['end']=cnt_frame
                    obj['box']=newbox
                    found = True
                    break
            if found == False:
                new_result_info.append({'start':cnt_frame,'end':cnt_frame,'box':newbox,'new':1})
    
    for obj in result_info:
        if cnt_frame-obj['end'] > int(fps*2):
            last_visit = (obj['end']-obj['start'])/fps
            break
        if obj['new']==1 and obj['end']-obj['start'] > int(fps):
            obj['new']=0
            newObj += 1
        new_result_info.append(obj)
    return newObj, new_result_info, last_visit

def process_faster_rcnn_result(result, args, cnt_frame, fps, result_info):
    newObj = 0
    new_result_info = []
    #print('raw:', result.shape)
    #print('raw:', result)
    ### filter label = person and probaibity > args.pt
    boxes = list(filter(lambda x: x[1] == 1 and args.prob_threshold < x[2], result[0][0]))
    result = sorted(result, key=lambda obj : (obj[1],obj[2]), reverse=True)
    for box in boxes:
        newbox = box[3:]
        #print(newbox)
        if len(result_info) == 0:
            new_result_info.append({'start':cnt_frame,'end':cnt_frame,'box':newbox,'new':1})
        else:
            found = False
            for i in range(len(result_info)):
                obj = result_info[i]
                if obj['end']==cnt_frame:
                    break
                iou = bb_intersection_over_union(obj['box'], newbox)
                #print('iou:',obj['box'], newbox, iou)
                if iou > args.prob_threshold:
                    obj['end']=cnt_frame
                    obj['box']=newbox
                    found = True
                    break
            if found == False:
                new_result_info.append({'start':cnt_frame,'end':cnt_frame,'box':newbox,'new':1})
    
    for obj in result_info:
        if cnt_frame-obj['end'] > int(fps):
            break
        if obj['new']==1 and obj['end']-obj['start'] > int(fps):
            obj['new']=0
            newObj += 1
        new_result_info.append(obj)
    return newObj, new_result_info

def process_yolo_result(result, args, cnt_frame, fps, result_info):
    newObj = 0
    new_result_info = []
    #print('raw:', result.shape)
    #print('raw:', result)
    result = sorted(result, key=lambda obj : (obj[1],obj[2]), reverse=True)
    #print(result)
    for i in range(len(result)):
        if result[i][2] == 0:
            continue
        for j in range(i + 1, len(result)):
            iou = bb_intersection_over_union(result[i][3:], result[j][3:])
            #print('iou', i, j, iou)
            if iou > 0.4:
                result[j][2] = 0
    result = [obj for obj in objects if obj[2] >= args.prob_threshold]
    for box in result:
        if box[1]!=0:
            continue
        print('each:', box)
        conf = box[2]
        if conf >= args.prob_threshold:
            newbox = box[3:]
            #print(newbox)
            if len(result_info) == 0:
                new_result_info.append({'start':cnt_frame,'end':cnt_frame,'box':newbox,'new':1})
            else:
                found = False
                for i in range(len(result_info)):
                    obj = result_info[i]
                    if obj['end']==cnt_frame:
                        break
                    iou = bb_intersection_over_union(obj['box'],newbox)
                    #print('iou:',obj['box'], newbox, iou)
                    if iou > args.prob_threshold:
                        obj['end']=cnt_frame
                        obj['box']=newbox
                        found = True
                        break
                if found == False:
                    new_result_info.append({'start':cnt_frame,'end':cnt_frame,'box':newbox,'new':1})
    
    for obj in result_info:
        if cnt_frame-obj['end'] > int(fps/2):
            break
        if obj['new']==1 and obj['end']-obj['start'] > int(fps/2):
            obj['new']=0
            newObj += 1
        new_result_info.append(obj)
    return newObj, new_result_info

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()
    net_second_input_shape = infer_network.get_second_input_shape()


    ### TODO: Handle the input stream ###
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #print('model', args.model)
    #print('video', height, width, fps)
    # if net_second_input_shape == None:
    #     print('input', net_input_shape)
    # else:
    #     print('input', net_second_input_shape)

    out_file_name = os.path.join('out',os.path.splitext(os.path.split(args.model)[1])[0]+'.mp4')
    out = cv2.VideoWriter(out_file_name, cv2.VideoWriter_fourcc(*'mp4v'), fps*3, (width,height))

    cnt_frame = 0
    cnt_total = 0
    total_duration = 0
    avg_duration = 0
    result_info = []

    start = time.time()
    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        #print('input', net_input_shape)
        #print('input2', net_second_input_shape)
        ### TODO: Pre-process the image as needed ###
        input_shape = net_input_shape
        if net_second_input_shape != None:
            input_shape = net_second_input_shape
        _,_,h,w = input_shape
        info = (height,width,1)
        #print(info)
        p_frame = cv2.resize(frame, (w, h), interpolation = cv2.INTER_CUBIC)
        p_frame = p_frame.transpose((2,0,1)) # Change data layout from HWC to CHW
        p_frame = p_frame.reshape(1, *p_frame.shape)

        ### TODO: Start asynchronous inference for specified request ###
        if net_second_input_shape != None:
            infer_network.exec_net(p_frame, info)
        else:
            infer_network.exec_net(p_frame, None)

        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
            
            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output()
            # output = infer_network.get_all_output()
            # result = list()
            # for layer_name, out_blob in output.items():
            #         #print('outputLayer', layer_name, out_blob.shape)
            #         out_blob = out_blob.reshape(infer_network.layers[infer_network.layers[layer_name].parents[0]].out_data[0].shape)
            #         #print('outputLayer2', layer_name, out_blob.shape)
            #         layer_params = YoloParams(infer_network.layers[layer_name].params, out_blob.shape[2])
            #         #log.info("Layer {} parameters: ".format(layer_name))
            #         #layer_params.log_params()
            #         result += parse_yolo_region(out_blob, p_frame.shape[2:],
            #                                         frame.shape[:-1], layer_params,
            #                                         args.prob_threshold)
            #print(result)
            

            ### TODO: Extract any desired stats from the results ###
            #print(result)process_ssd_result
            last_visit = None
            newObj, result_info, last_visit = process_ssd_result(result, args, cnt_frame, fps, result_info)
            #newObj, result_info = process_faster_rcnn_result(result, args, cnt_frame, fps, result_info)
            #newObj, result_info = process_yolo_result(objects, args, cnt_frame, fps, result_info)
            cnt_total += newObj
            cnt_current = len(result_info)
            total_duration += cnt_current
            if cnt_total != 0:
                avg_duration = total_duration / cnt_total / fps
            print('{} result: {}, {}, {}, {}s'.format(cnt_frame,cnt_total, newObj, cnt_current, avg_duration))
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            if client != None:
                client.publish("person", json.dumps({"count": cnt_current,"total": cnt_total}))
                client.publish("person/duration", json.dumps({"duration": avg_duration}))
            #frame = draw_boxes(frame, result, args, width, height)
            out_frame = draw_result_boxes(frame, result_info, width, height)
            frame=out_frame

        cnt_frame += 1

        ### TODO: Send the frame to the FFMPEG server ###
        #sys.stdout.buffer.write(out_frame)
        #sys.stdout.flush()

        ### TODO: Write an output image if `single_image_mode` ###
        # Write out the frame
        out.write(frame)
    print("it took", time.time() - start, "seconds.")
    out.release()
    cap.release()
    cv2.destroyAllWindows()


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    #client = connect_mqtt()
    client = None
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
