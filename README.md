# **Working Attendance with AI Face Identification**

## **Overall**
The Attendence System with tablet (phrase 1) shorted as CIOS is created in order to manage the labor productivity of a company or an organization...  

To run any prefered version of the project below on the local machine requires the following libraries and dependences:  

<table>
    <tr>
     <th>No</th>
     <th>Library/Driver</th>
     <th>Tensorflow-Ver</th>
     <th>Pytorch-Ver</th>
    </tr>
    <tr>
     <th>1</td>
     <td></td>
     <td>Tensorflow: 2.20.0</td>
     <td>torch: 2.2.2  torchvision: 0.17.2</td>
    </tr>
    <tr>
     <th>2</td>
     <td>flask</td>
     <td colspan="2">3.1.2</td>
    </tr>
    <tr>
     <th>3</td>
     <td>scipy</td>
     <td>latest</td>
     <td>1.15.3</td>
    </tr>
    <tr>
     <th>4</td>
     <td>opencv-python (cv2 - supported by libgl1 and libglib2.0-0)</td>
     <td>4.12.0</td>
     <td>4.9.0.80</td>
    </tr>
    <tr>
     <th>5</td>
     <td>numpy</td>
     <td>2.2.6</td>
     <td>1.26.4</td>
    </tr>
    <tr>
     <th>6</td>
     <td>joblib</td>
     <td colspan="2"><1.5.2></td>
    </tr>
    <tr>
     <th>7</td>
     <td></td>
     <td>keras-facenet: 0.3.2</td>
     <td>facenet-pytorch: 2.6.0</td>
    </tr>
    <tr>
     <th>8</td>
     <td>mtcnn</td>
     <td colspan="2">1.0.0</td>
    </tr>
    <tr>
     <th>9</td>
     <td>scikit-learn</td>
     <td colspan="2">1.7.2</td>
    </tr>
    <tr>
     <th>10</td>
     <td>cuDNN</td>
     <td> v91301</td>
     <td>none</td>
    </tr>
    <tr>
     <th>11</td>
     <td>GPU NVDIDIA CUDA driver (toolkit)</td>
     <td>12.2</td>
     <td>12.4</td>
    </tr>
    <tr>
     <th>12</td>
     <td>GPU NVIDIA driver</td>
     <td>530.x</td>
     <td>550.x</td>
    </tr>
    <tr>
     <th>13</td>
     <td>python</td>
     <td>3.12.x</td>
     <td>3.10.x</td>
    </tr>
</table>
 
**_Note1: If there are any changes in those above versions, please ensure the compatibility among them._**  

## **System details**
**1. About core technology**

* MTCNN is used for Face detection
* FACENET is use for generating embedding vector
* The AI inference applies the Cosine Similarity 

**2. About the system mechanism**
* Input image is retrieved from tablet camera through WEB UI (with size of 1280x960 pixels )
* The AI will handle the face in the image and return a result
* The result is updated on google sheet and noticed in Telegram chatbot (optional)
* If the result is unknown person, restore the images

**3. About data and AI**
* Raw dataset is originally collected through short videos (include glasses but no masks) given by each company's member.
* MTCNN and FACENET are combined with the similarity to train with the processed dataset, then finally generate the AI - an new specific model.

**4. System Review (Tensorflow-Ver)**  

|Sections|Details|  
|-----------|--------|  
|Pre-process data (videos) | Extract about 80 images per video |  
|Refine data| Sharpen, covert to RGB, get box and resize to 160x160 pixels |  
|Fine-tuning and training| Evaluate during training with acccuracy, loss, F1 score, overfit... |  

|AI specifications | Point|
|----|----|
|Accuracy|0.98|
|Macro average|0.98|
|Weighted average |0.97|
|Test loss| 0.4|  

| System Processing Duration| Time (second)|  
|----|----|
|Whole process (include data transport)| around 0.9s |
|image convert| about 0.4s|
|AI Inference| about 0.2s |
|Google Sheet update | about 1.x ~ 2.0s |  

**_Note2: The pytorch-Ver is about 0.1s slower in the first inference and about 0.2s in total without cuDNN_**  
**_Note3: There are no differences in mechanism, dataset, model specifications between both system versions except performance_**