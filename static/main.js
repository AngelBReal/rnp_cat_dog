const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const resultado = document.getElementById("resultado");

let facingMode = "user"; // inicia con la cámara frontal
let stream = null;

function iniciarCamara() {
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
  }

  navigator.mediaDevices.getUserMedia({
    video: { facingMode: facingMode },
    audio: false
  })
  .then(mediaStream => {
    stream = mediaStream;
    const video = document.getElementById("video");
    video.srcObject = stream;
  })
  .catch(err => {
    console.error("Error al acceder a la cámara:", err);
  });
}

function cambiarCamara() {
  facingMode = (facingMode === "user") ? "environment" : "user";
  iniciarCamara();
}


navigator.mediaDevices.getUserMedia({ video: true })
  .then((stream) => {
    video.srcObject = stream;
    video.play();
    setInterval(capturarYPredecir, 1000);
  })
  .catch((err) => {
    console.error("No se pudo acceder a la cámara", err);
  });

function capturarYPredecir() {
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const dataUrl = canvas.toDataURL("image/jpeg");
  const modelo = document.getElementById("modelo").value;

  fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: dataUrl, model: modelo })
  })
    .then(res => res.json())
    .then(data => {
      resultado.textContent = `Resultado: ${data.resultado}`;
    })
    .catch(err => {
      console.error("Error al predecir", err);
    });
}

window.onload = iniciarCamara;