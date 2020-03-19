const MODEL_URL = "https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_140_224/classification/3/default/1";
const DICT_URL = "https://cors-anywhere.herokuapp.com/https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt";

// Global
let model;
let dictionary;
let webWorker = null;
let workerModelIsReady = false;
let isWaiting = false;

// DOM Elt References
var DOM_EL = {
  canvas: null,
  ctx: null,
  video: null,
}

window.addEventListener('DOMContentLoaded', () => {
  // assign references
  DOM_EL.video = document.getElementById("video");
  DOM_EL.canvas = document.getElementById("canvas");
  DOM_EL.canvas.width = window.innerWidth;
  DOM_EL.canvas.height = window.innerWidth;
  DOM_EL.ctx = DOM_EL.canvas.getContext("2d", { alpha: false, desynchronized: false });

  init();
});

const init = async function() {
  try {
    DOM_EL.video.srcObject = await setupCamera();
  } catch(err) {
    console.log(err);
  }
  setupModel();

  DOM_EL.video.onloadeddata = e => {
    DOM_EL.video.play();
    render();
    if (window.Worker) {
      setInterval(offloadPredict, 1000);
    } else {
      setInterval(predict, 2000);
    }
  }
}

const setupCamera = async function() {
  return navigator.mediaDevices
  .getUserMedia({ video: { facingMode: "environment" }, audio: false })
  .then(stream => stream)
  .catch(function(error) {
    console.error("Oops. Something is broken.", error);
  });
}

const setupModel = async function() {
  if (window.Worker) {
    webWorker = new Worker('web-worker.js');
    webWorker.onmessage = evt =>{
      isWaiting = !isWaiting;
      if (evt.modelIsReady) {
        workerModelIsReady = true;
      }
      console.log(evt.data);
    }

    return;
  } else {
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
  }
}

const predict = async function() {
  if (!workerModelIsReady) {
    return;
  }
  const scores = tf.tidy(() => {
    const imgAsTensor = tf.browser.fromPixels(DOM_EL.canvas);
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

  // predicted class
  console.log(prediction.label);
  // probability
  console.log(parseFloat(prediction.prob.toFixed(2)));
}

const offloadPredict = async function() {
  if (!isWaiting && !workerModelIsReady) {
    const imageData = DOM_EL.ctx.getImageData(0, 0, DOM_EL.canvas.width, DOM_EL.canvas.height);
    console.log(imageData);
    webWorker.postMessage(imageData);
    isWaiting = !isWaiting;
  }
}

const render = function() {
  DOM_EL.ctx.drawImage(DOM_EL.video, 0, 0, window.innerWidth, window.innerWidth);
  window.requestAnimationFrame(render);
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