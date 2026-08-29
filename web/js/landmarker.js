/**
 * landmarker.js
 * Camera + MediaPipe HandLandmarker (WebAssembly, runs locally in the browser).
 * Draws the hand skeleton on a canvas over the video.
 */

import { FilesetResolver, HandLandmarker } from
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

/** Point pairs that form the hand skeleton (standard MediaPipe topology). */
export const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],        // thumb
  [0,5],[5,6],[6,7],[7,8],        // index finger
  [5,9],[9,10],[10,11],[11,12],   // middle finger
  [9,13],[13,14],[14,15],[15,16], // ring finger
  [13,17],[17,18],[18,19],[19,20],// pinky
  [0,17]                          // palm edge
];

export async function createLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
  return HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.6,
    minHandPresenceConfidence: 0.6,
    minTrackingConfidence: 0.6,
  });
}

export async function startCamera(videoEl) {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
    audio: false,
  });
  videoEl.srcObject = stream;
  await new Promise(res => (videoEl.onloadedmetadata = res));
  await videoEl.play();
  return stream;
}

/** Draws the skeleton on a canvas. The canvas is CSS-mirrored just like the video,
 *  so coordinates are drawn without additional flipping. */
export function drawSkeleton(ctx, landmarks, w, h, color = "#1D6FE8") {
  ctx.clearRect(0, 0, w, h);
  if (!landmarks) return;
  ctx.lineWidth = 3;
  ctx.strokeStyle = color;
  ctx.lineCap = "round";
  for (const [a, b] of HAND_CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(landmarks[a].x * w, landmarks[a].y * h);
    ctx.lineTo(landmarks[b].x * w, landmarks[b].y * h);
    ctx.stroke();
  }
  ctx.fillStyle = "#0F2A46";
  for (const p of landmarks) {
    ctx.beginPath();
    ctx.arc(p.x * w, p.y * h, 4, 0, Math.PI * 2);
    ctx.fill();
  }
}
