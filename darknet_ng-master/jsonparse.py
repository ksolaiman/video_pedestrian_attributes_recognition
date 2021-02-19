import os
import json

number_of_images_with_more_than_1_person = 0
number_of_images_with_0_person = 0

entries = os.listdir('output_json_queryloader_sized/')
jsonpath = "/homes/ksolaima/scratch1/yuange250_video_pedestrian_attributes_recognition-master/darknet_ng-master/output_json_queryloader_sized/"
output_color_file = open("color_sampling_outputs.txt", "a")
img_file = open("../test_frames.txt", "r")
for file in img_file:
    jsonname = file.split("/")[-1][:-5]
#for jsonname in entries:
    jsonfile = jsonpath + jsonname + "_0.json"
    with open(jsonfile, encoding="ISO-8859-1") as f:
        data = json.load(f) 
        # Iterating through the json 
        person_num = len(data) 
        if person_num > 1:
            number_of_images_with_more_than_1_person += 1
        elif person_num == 0:
            number_of_images_with_more_than_1_person += 0
        
        print("\n")
        area = 0
        for i in range(1,person_num+1):
            personID = "person " + str(i)
            if int(data[personID]["size"]) > area:
                area = int(data[personID]["size"])
                person_code = data[personID]["color_code"] 
                pid = i
                head_code = int(person_code[0])
                upper_code = int(person_code[1])
                bottom_code = int(person_code[2])
            print(personID, pid, head_code, upper_code, bottom_code)
        output_color_file.write(jsonname + "\t"+ str(upper_code) +"\t"+ str(bottom_code) + "\n")
output_color_file.close()
        
        
print(number_of_images_with_more_than_1_person)
print(number_of_images_with_0_person)
