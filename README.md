# FaceId:
tensorflow FaceId（DeepFace）

## DataSet：
python tools/preprocess_face_data.py


Data:

  --train:
  
  ----:face_one
  
  ----:face_two
.
.
.

  --test:...

## Train:
python tools/train.py


## Test:
python tools/test.py

acc:0.91 step:10000

## Save:
python tools/save.py

Save the face feature verctor 

tools/compare.py
compare the face to the database
