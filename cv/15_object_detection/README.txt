https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md


git clone https://github.com/tensorflow/models.git

cd models/research
# Compile protos.
protoc object_detection/protos/*.proto --python_out=.
# Install TensorFlow Object Detection API.
cp object_detection/packages/tf2/setup.py .
python -m pip install --use-feature=2020-resolver .

# Test the installation.
python object_detection/builders/model_builder_tf2_test.py




slim文件夹是给tf1用的，tf2不用。

tutorial是转换而来，tf2实现的。



https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md


