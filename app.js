const MODEL_PATH = "models/best.onnx";
const MODEL_SIZE = 640;

let session = null;
let running = false;
let animationId = null;
let confidenceThreshold = 0.35;

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const video = document.getElementById("video");
const image = document.getElementById("image");

const statusText = document.getElementById("status");
const countText = document.getElementById("count");

const imageUpload = document.getElementById("imageUpload");
const videoUpload = document.getElementById("videoUpload");
const webcamBtn = document.getElementById("webcamBtn");
const stopBtn = document.getElementById("stopBtn");

const confSlider = document.getElementById("confSlider");
const confValue = document.getElementById("confValue");

confSlider.addEventListener("input", () => {
  confidenceThreshold = parseFloat(confSlider.value);
  confValue.innerText = confidenceThreshold.toFixed(2);
});

async function loadModel() {
  try {
    statusText.innerText = "Loading model...";

    session = await ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ["wasm"]
    });

    statusText.innerText = "Model loaded successfully";
  } catch (error) {
    console.error(error);
    statusText.innerText = "Failed to load model";
  }
}

loadModel();

imageUpload.addEventListener("change", async function (event) {
  stopProcessing();

  const file = event.target.files[0];
  if (!file) return;

  const url = URL.createObjectURL(file);
  image.src = url;

  image.onload = async () => {
    canvas.width = image.width;
    canvas.height = image.height;

    ctx.drawImage(image, 0, 0);

    statusText.innerText = "Running detection on image...";
    const detections = await detect(image);

    drawImageWithDetections(image, detections);
    statusText.innerText = "Image detection completed";
  };
});

videoUpload.addEventListener("change", function (event) {
  stopProcessing();

  const file = event.target.files[0];
  if (!file) return;

  const url = URL.createObjectURL(file);
  video.srcObject = null;
  video.src = url;
  video.hidden = true;

  video.onloadedmetadata = () => {
    video.play();
    running = true;
    statusText.innerText = "Running detection on video...";
    processVideo();
  };
});

webcamBtn.addEventListener("click", async function () {
  stopProcessing();

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false
    });

    video.srcObject = stream;
    video.hidden = true;
    video.play();

    running = true;
    statusText.innerText = "Webcam started";
    processVideo();
  } catch (error) {
    console.error(error);
    statusText.innerText = "Unable to access webcam";
  }
});

stopBtn.addEventListener("click", stopProcessing);

function stopProcessing() {
  running = false;

  if (animationId) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }

  if (video.srcObject) {
    video.srcObject.getTracks().forEach(track => track.stop());
    video.srcObject = null;
  }

  video.pause();
  video.removeAttribute("src");

  statusText.innerText = session ? "Ready" : "Loading model...";
}

async function processVideo() {
  if (!running) return;

  if (video.readyState >= 2) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const detections = await detect(video);
    drawImageWithDetections(video, detections);
  }

  animationId = requestAnimationFrame(processVideo);
}

async function detect(source) {
  if (!session) {
    alert("Model is still loading. Please wait.");
    return [];
  }

  const inputTensor = preprocess(source);

  const feeds = {};
  feeds[session.inputNames[0]] = inputTensor;

  const results = await session.run(feeds);
  const output = results[session.outputNames[0]];

  const detections = postprocess(
    output,
    source.width || source.videoWidth,
    source.height || source.videoHeight
  );

  return detections;
}

function preprocess(source) {
  const tempCanvas = document.createElement("canvas");
  const tempCtx = tempCanvas.getContext("2d");

  tempCanvas.width = MODEL_SIZE;
  tempCanvas.height = MODEL_SIZE;

  const sourceWidth = source.width || source.videoWidth;
  const sourceHeight = source.height || source.videoHeight;

  const scale = Math.min(MODEL_SIZE / sourceWidth, MODEL_SIZE / sourceHeight);
  const newWidth = Math.round(sourceWidth * scale);
  const newHeight = Math.round(sourceHeight * scale);

  const padX = Math.floor((MODEL_SIZE - newWidth) / 2);
  const padY = Math.floor((MODEL_SIZE - newHeight) / 2);

  tempCtx.fillStyle = "rgb(114,114,114)";
  tempCtx.fillRect(0, 0, MODEL_SIZE, MODEL_SIZE);
  tempCtx.drawImage(source, padX, padY, newWidth, newHeight);

  const imageData = tempCtx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE);
  const data = imageData.data;

  const float32Data = new Float32Array(3 * MODEL_SIZE * MODEL_SIZE);

  for (let i = 0; i < MODEL_SIZE * MODEL_SIZE; i++) {
    float32Data[i] = data[i * 4] / 255.0;
    float32Data[i + MODEL_SIZE * MODEL_SIZE] = data[i * 4 + 1] / 255.0;
    float32Data[i + 2 * MODEL_SIZE * MODEL_SIZE] = data[i * 4 + 2] / 255.0;
  }

  return new ort.Tensor("float32", float32Data, [1, 3, MODEL_SIZE, MODEL_SIZE]);
}

function postprocess(output, originalWidth, originalHeight) {
  const data = output.data;
  const dims = output.dims;

  let rows;
  let cols;
  let transposed = false;

  if (dims.length === 3) {
    if (dims[1] < dims[2]) {
      cols = dims[1];
      rows = dims[2];
      transposed = true;
    } else {
      rows = dims[1];
      cols = dims[2];
    }
  } else {
    console.error("Unsupported output shape:", dims);
    return [];
  }

  const boxes = [];

  const scale = Math.min(MODEL_SIZE / originalWidth, MODEL_SIZE / originalHeight);
  const newWidth = originalWidth * scale;
  const newHeight = originalHeight * scale;
  const padX = (MODEL_SIZE - newWidth) / 2;
  const padY = (MODEL_SIZE - newHeight) / 2;

  for (let i = 0; i < rows; i++) {
    let cx, cy, w, h, score;

    if (transposed) {
      cx = data[0 * rows + i];
      cy = data[1 * rows + i];
      w = data[2 * rows + i];
      h = data[3 * rows + i];
      score = data[4 * rows + i];
    } else {
      cx = data[i * cols + 0];
      cy = data[i * cols + 1];
      w = data[i * cols + 2];
      h = data[i * cols + 3];
      score = data[i * cols + 4];
    }

    if (score < confidenceThreshold) continue;

    let x1 = cx - w / 2;
    let y1 = cy - h / 2;
    let x2 = cx + w / 2;
    let y2 = cy + h / 2;

    x1 = (x1 - padX) / scale;
    y1 = (y1 - padY) / scale;
    x2 = (x2 - padX) / scale;
    y2 = (y2 - padY) / scale;

    x1 = Math.max(0, Math.min(originalWidth, x1));
    y1 = Math.max(0, Math.min(originalHeight, y1));
    x2 = Math.max(0, Math.min(originalWidth, x2));
    y2 = Math.max(0, Math.min(originalHeight, y2));

    boxes.push({
      x1,
      y1,
      x2,
      y2,
      score
    });
  }

  return nonMaxSuppression(boxes, 0.45);
}

function nonMaxSuppression(boxes, iouThreshold) {
  boxes.sort((a, b) => b.score - a.score);

  const selected = [];

  while (boxes.length > 0) {
    const current = boxes.shift();
    selected.push(current);

    boxes = boxes.filter(box => calculateIoU(current, box) < iouThreshold);
  }

  return selected;
}

function calculateIoU(a, b) {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);

  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);

  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);

  return intersection / (areaA + areaB - intersection + 1e-6);
}

function drawImageWithDetections(source, detections) {
  const width = source.width || source.videoWidth;
  const height = source.height || source.videoHeight;

  canvas.width = width;
  canvas.height = height;

  ctx.drawImage(source, 0, 0, width, height);

  detections.forEach(det => {
    const boxWidth = det.x2 - det.x1;
    const boxHeight = det.y2 - det.y1;

    ctx.strokeStyle = "#00ff66";
    ctx.lineWidth = Math.max(2, width / 300);
    ctx.strokeRect(det.x1, det.y1, boxWidth, boxHeight);

    const label = `License Plate ${(det.score * 100).toFixed(1)}%`;

    ctx.font = `${Math.max(14, width / 45)}px Arial`;
    const textWidth = ctx.measureText(label).width;

    ctx.fillStyle = "#00ff66";
    ctx.fillRect(det.x1, det.y1 - 28, textWidth + 12, 28);

    ctx.fillStyle = "#000000";
    ctx.fillText(label, det.x1 + 6, det.y1 - 8);
  });

  countText.innerText = detections.length;
}
