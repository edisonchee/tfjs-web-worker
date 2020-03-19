importScripts('https://unpkg.com/@tensorflow/tfjs');
tf.setBackend('cpu');

const MODEL_URL = "https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_140_224/classification/3/default/1";
const DICT_URL = "https://cors-anywhere.herokuapp.com/https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt";

let model;
let dictionary;

const setup = async () => {
  try {
    model = await tf.loadGraphModel(MODEL_URL, { fromTFHub: true });
    const response = await tf.util.fetch(DICT_URL);
    const text = await response.text();
    dictionary = text.trim().split('\n');
  } catch(err) {
    console.error("Can't load model: ", err)
  }
  
  const zeros = tf.zeros([1, 224, 224, 3]);
  // warm-up the model
  model.predict(zeros);
  postMessage({ modelIsReady: true});
}

setup();

onmessage = evt => {
  if (model) {
    predict(evt.data);
  }
}

const predict = async function(imageData) {
  const scores = tf.tidy(() => {
    const imgAsTensor = tf.browser.fromPixels(imageData);
    const centerCroppedImg = centerCropAndResize(imgAsTensor);
    const processedImg = centerCroppedImg.div(127.5).sub(1);
    return model.predict(processedImg);
  })

  const probabilities = await scores.data();
  scores.dispose();
  const result = Array.from(probabilities)
                     .map((prob, i) => ({label: dictionary[i], prob}));

  const prediction = result.reduce(function(prev, current) {
    return (prev.prob > current.prob) ? prev : current
  })

  postMessage([prediction.label, parseFloat(prediction.prob.toFixed(2))]);
}

function centerCropAndResize(img) {
  return tf.tidy(() => {
    const [height, width] = img.shape.slice(0, 2);
    let top = 0;
    let left = 0;
    if (height > width) {
      top = (height - width) / 2;
    } else {
      left = (width - height) / 2;
    }
    const size = Math.min(width, height);
    const boxes = [
      [top / height, left / width, (top + size) / height, (left + size) / width]
    ];
    const boxIndices = [0];
    return tf.image.cropAndResize(
        img.toFloat().expandDims(), 
        boxes, 
        boxIndices, 
        [224, 224]
    );
  });
}