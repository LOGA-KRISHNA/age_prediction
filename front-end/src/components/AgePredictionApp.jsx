import React, { useState, useRef, useEffect } from "react";
import {
  Upload,
  Camera,
  StopCircle,
  PlayCircle,
  Loader2,
} from "lucide-react";

export default function AgePredictionApp() {
  const [image, setImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [realTimePredictions, setRealTimePredictions] = useState([]);
  const [error, setError] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const [isRealTimeMode, setIsRealTimeMode] = useState(false);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);

  // Start webcam
  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
      });
      streamRef.current = stream;
      setIsWebcamActive(true);
    } catch (err) {
      setError("Failed to access webcam: " + err.message);
    }
  };

  // Attach stream after video element is rendered
  useEffect(() => {
    if (isWebcamActive && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current;
      videoRef.current.play().catch(() => {});
    }
  }, [isWebcamActive]);

  // Stop webcam
  const stopWebcam = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
    setIsWebcamActive(false);
    stopRealTime();
  };

  // Upload image
  const handleUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onloadend = () => {
      setImage(reader.result);
      sendToBackend(reader.result);
    };
    reader.readAsDataURL(file);
  };

  // Capture single frame
  const captureImage = () => {
    if (!videoRef.current || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    ctx.drawImage(videoRef.current, 0, 0);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.8);
    setImage(dataUrl);
    sendToBackend(dataUrl);
  };

  // Send image to backend
  const sendToBackend = async (base64Image) => {
    setIsProcessing(true);
    setError(null);
    setPrediction(null);
    try {
      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "ngrok-skip-browser-warning": "true",
        },
        body: JSON.stringify({ image: base64Image }),
      });
      if (!res.ok) {
        let message = `Request failed with status ${res.status}`;
        try {
          const errJson = await res.json();
          message = errJson.error || errJson.detail || message;
        } catch {}
        throw new Error(message);
      }
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setPrediction(Math.round(data.age * 10) / 10);
      if (isRealTimeMode) {
        const newEntry = {
          age: Math.round(data.age * 10) / 10,
          time: new Date().toLocaleTimeString(),
        };
        setRealTimePredictions((prev) => [newEntry, ...prev].slice(0, 5));
      }
    } catch (err) {
      setError("Prediction failed: " + err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  // Real-time loop
  const startRealTime = () => {
    if (!isWebcamActive) return;
    setIsRealTimeMode(true);
    intervalRef.current = setInterval(() => {
      captureImage();
    }, 3000);
  };

  const stopRealTime = () => {
    setIsRealTimeMode(false);
    if (intervalRef.current) clearInterval(intervalRef.current);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 to-blue-100 flex flex-col items-center p-6">
      <h1 className="text-3xl font-bold mb-6 text-indigo-700">
        Age Prediction AI
      </h1>

      {/* Upload Section */}
      <div className="bg-white shadow-md rounded-xl p-6 mb-6 w-full max-w-md text-center">
        <h2 className="text-lg font-semibold mb-4">Upload Image</h2>
        <label className="flex flex-col items-center justify-center border-2 border-dashed border-indigo-300 rounded-lg p-6 cursor-pointer hover:bg-indigo-50">
          <Upload className="w-10 h-10 text-indigo-500 mb-2" />
          <span className="text-indigo-600 font-medium">Click to Upload</span>
          <input
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleUpload}
          />
        </label>
      </div>

      {/* Webcam Section */}
      <div className="bg-white shadow-md rounded-xl p-6 mb-6 w-full max-w-md text-center">
        <h2 className="text-lg font-semibold mb-4">Webcam</h2>
        {!isWebcamActive ? (
          <button
            onClick={startWebcam}
            className="px-4 py-2 bg-indigo-600 text-white rounded-lg flex items-center gap-2 mx-auto"
          >
            <Camera className="w-5 h-5" /> Start Webcam
          </button>
        ) : (
          <>
            <video
              ref={videoRef}
              autoPlay
              muted
              playsInline
              className="w-full rounded-lg border mb-4"
            />
            <canvas ref={canvasRef} className="hidden" />
            <div className="flex gap-2 justify-center">
              <button
                onClick={captureImage}
                className="px-4 py-2 bg-green-500 text-white rounded-lg"
              >
                Capture
              </button>
              {!isRealTimeMode ? (
                <button
                  onClick={startRealTime}
                  className="px-4 py-2 bg-blue-500 text-white rounded-lg flex items-center gap-2"
                >
                  <PlayCircle className="w-5 h-5" /> Real-Time
                </button>
              ) : (
                <button
                  onClick={stopRealTime}
                  className="px-4 py-2 bg-red-500 text-white rounded-lg flex items-center gap-2"
                >
                  <StopCircle className="w-5 h-5" /> Stop
                </button>
              )}
              <button
                onClick={stopWebcam}
                className="px-4 py-2 bg-gray-500 text-white rounded-lg"
              >
                Stop Webcam
              </button>
            </div>
          </>
        )}
      </div>

      {/* Results */}
      <div className="bg-white shadow-md rounded-xl p-6 w-full max-w-md text-center">
        <h2 className="text-lg font-semibold mb-4">Result</h2>
        {isProcessing && (
          <div className="flex justify-center items-center gap-2 text-indigo-600">
            <Loader2 className="w-5 h-5 animate-spin" /> Processing...
          </div>
        )}
        {prediction && !isRealTimeMode && (
          <p className="text-xl font-bold text-green-600">
            {prediction} years old
          </p>
        )}
        {error && <p className="text-red-500">{error}</p>}
        {isRealTimeMode && realTimePredictions.length > 0 && (
          <div className="text-left">
            <h3 className="font-medium mb-2">Recent Predictions:</h3>
            <ul className="space-y-1">
              {realTimePredictions.map((p, idx) => (
                <li key={idx} className="text-gray-700">
                  {p.age} yrs <span className="text-sm">({p.time})</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Preview */}
      {image && (
        <div className="mt-6">
          <h2 className="text-lg font-semibold mb-2">Preview</h2>
          <img
            src={image}
            alt="preview"
            className="w-64 rounded-lg shadow-md border"
          />
        </div>
      )}
    </div>
  );
}
