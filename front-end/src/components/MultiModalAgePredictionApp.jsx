import React, { useState, useRef, useEffect } from "react";

export default function MultiModalAgePredictionApp() {
  // Image states
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  
  // Audio states
  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [audioFileName, setAudioFileName] = useState(null);
  
  // Prediction states
  const [predictions, setPredictions] = useState({
    image: null,
    audio: null,
    combined: null
  });
  const [error, setError] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [hasInputs, setHasInputs] = useState(false);

  // Refs
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const recordingTimerRef = useRef(null);

  // Check if we have any inputs
  useEffect(() => {
    setHasInputs(!!image || !!audioBlob);
  }, [image, audioBlob]);

  // === IMAGE HANDLING ===

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
      videoRef.current.play().catch((e) => console.error("Video play failed:", e));
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
  };

  // Upload image
  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onloadend = () => {
      setImage(reader.result);
      setImagePreview(reader.result);
    };
    reader.readAsDataURL(file);
  };

  // Capture image from webcam
  const captureImage = () => {
    if (!videoRef.current || !canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    ctx.drawImage(videoRef.current, 0, 0);
    
    const dataUrl = canvas.toDataURL("image/jpeg", 0.8);
    setImage(dataUrl);
    setImagePreview(dataUrl);
    stopWebcam();
  };

  // Clear image
  const clearImage = () => {
    setImage(null);
    setImagePreview(null);
    setPredictions(prev => ({ ...prev, image: null, combined: null }));
  };

  // === AUDIO HANDLING ===

  // Handle audio file upload
  const handleAudioUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const allowedTypes = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/m4a', 'audio/webm', 'audio/ogg', 'audio/flac'];
    if (!allowedTypes.includes(file.type) && !file.name.match(/\.(wav|mp3|m4a|webm|ogg|flac)$/i)) {
      setError("Please upload a valid audio file (WAV, MP3, M4A, WebM, OGG, or FLAC)");
      return;
    }

    if (file.size > 50 * 1024 * 1024) {
      setError("Audio file too large. Please upload a file smaller than 50MB");
      return;
    }

    // The File object is already a Blob. No need to re-read it.
    setAudioBlob(file);
    setAudioUrl(URL.createObjectURL(file));
    setAudioFileName(file.name);
    setRecordingDuration(0);
    setError(null);
  };

  // Start recording audio
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });
      
      mediaRecorderRef.current = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });
      audioChunksRef.current = [];
      
      mediaRecorderRef.current.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };
      
      mediaRecorderRef.current.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { 
          type: 'audio/webm;codecs=opus' 
        });
        setAudioBlob(audioBlob);
        setAudioUrl(URL.createObjectURL(audioBlob));
        setAudioFileName(null); 
        
        stream.getTracks().forEach(track => track.stop());
      };
      
      mediaRecorderRef.current.start();
      setIsRecording(true);
      setRecordingDuration(0);
      
      recordingTimerRef.current = setInterval(() => {
        setRecordingDuration(prev => prev + 1);
      }, 1000);
      
    } catch (err) {
      setError("Failed to access microphone: " + err.message);
    }
  };

  // Stop recording
  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current);
        recordingTimerRef.current = null;
      }
    }
  };

  // Clear audio
  const clearAudio = () => {
    setAudioBlob(null);
    setAudioUrl(null);
    setRecordingDuration(0);
    setAudioFileName(null);
    setPredictions(prev => ({ ...prev, audio: null, combined: null }));
    
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
    }
  };

  // === PREDICTION HANDLING ===

  // Convert blob to base64
  const blobToBase64 = (blob) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  };

  // Send prediction request
  const predictAge = async () => {
    if (!image && !audioBlob) {
      setError("Please provide at least an image or audio recording");
      return;
    }

    setIsProcessing(true);
    setError(null);
    setPredictions({ image: null, audio: null, combined: null });

    try {
      const requestData = {};
      
      if (image) {
        requestData.image = image;
      }
      
      if (audioBlob) {
        const audioBase64 = await blobToBase64(audioBlob);
        requestData.audio = audioBase64;
      }

      // NOTE: This fetch URL is for a local server. 
      // Replace with your actual API endpoint.
      const response = await fetch("http://localhost:8000/predict/multimodal", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestData),
      });

      if (!response.ok) {
        let message = `Request failed with status ${response.status}`;
        try {
          const errJson = await response.json();
          message = errJson.error || errJson.detail || message;
        } catch {}
        throw new Error(message);
      }

      const data = await response.json();
      
      if (data.error) {
        throw new Error(data.error);
      }

      const newPredictions = {};
      if (data.image_prediction) newPredictions.image = data.image_prediction;
      if (data.audio_prediction) newPredictions.audio = data.audio_prediction;
      if (data.combined_age !== null && data.combined_age !== undefined) {
        newPredictions.combined = data.combined_age;
      }
      
      setPredictions(newPredictions);

    } catch (err) {
      setError("Prediction failed: " + err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  // Format duration helper
  const formatDuration = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Cleanup function
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current);
      }
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }
    };
  }, [audioUrl]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 to-indigo-100 p-6 font-sans">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold text-center mb-8 text-indigo-800">
          Multi-Modal Age Prediction AI
        </h1>
        <p className="text-center text-gray-600 mb-8">
          Upload an image and/or record/upload audio to predict age using advanced AI models
        </p>

        {/* === INPUT SECTIONS === */}
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          {/* IMAGE INPUT SECTION */}
          <div className="bg-white shadow-lg rounded-xl p-6">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2 text-indigo-700">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" /></svg>
              Image Input
            </h2>
            {!imagePreview ? (
              <div className="space-y-4">
                <label className="flex flex-col items-center justify-center border-2 border-dashed border-indigo-300 rounded-lg p-6 cursor-pointer hover:bg-indigo-50 transition-colors">
                  <svg className="w-12 h-12 text-indigo-500 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" /></svg>
                  <span className="text-indigo-600 font-medium">Click to Upload Image</span>
                  <span className="text-sm text-gray-500">or drag and drop</span>
                  <input type="file" accept="image/*" className="hidden" onChange={handleImageUpload} />
                </label>
                <div className="text-center">
                  <div className="text-gray-500 mb-2">OR</div>
                  {!isWebcamActive ? (
                    <button onClick={startWebcam} className="px-4 py-2 bg-indigo-600 text-white rounded-lg flex items-center gap-2 mx-auto hover:bg-indigo-700 transition-colors">
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" /></svg>
                      Use Webcam
                    </button>
                  ) : (
                    <div className="space-y-3">
                      <video ref={videoRef} autoPlay muted playsInline className="w-full rounded-lg border" />
                      <canvas ref={canvasRef} className="hidden" />
                      <div className="flex gap-2 justify-center">
                        <button onClick={captureImage} className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors">Capture</button>
                        <button onClick={stopWebcam} className="px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors">Cancel</button>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <img src={imagePreview} alt="Selected" className="w-full rounded-lg shadow-md" />
                <div className="flex gap-2 justify-center">
                  <button onClick={clearImage} className="px-4 py-2 bg-red-500 text-white rounded-lg flex items-center gap-2 hover:bg-red-600 transition-colors">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg> Remove
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* AUDIO INPUT SECTION */}
          <div className="bg-white shadow-lg rounded-xl p-6">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2 text-indigo-700">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" /></svg>
              Audio Input
            </h2>
            {!audioBlob ? (
              <div className="space-y-4">
                <label className="flex flex-col items-center justify-center border-2 border-dashed border-indigo-300 rounded-lg p-6 cursor-pointer hover:bg-indigo-50 transition-colors">
                  <svg className="w-12 h-12 text-indigo-500 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" /></svg>
                  <span className="text-indigo-600 font-medium">Click to Upload Audio</span>
                  <span className="text-sm text-gray-500">WAV, MP3, M4A, etc. (max 50MB)</span>
                  <input type="file" accept="audio/*,.wav,.mp3,.m4a,.webm,.ogg,.flac" className="hidden" onChange={handleAudioUpload} />
                </label>
                <div className="text-center">
                  <div className="text-gray-500 mb-2">OR</div>
                  {!isRecording ? (
                    <button onClick={startRecording} className="px-6 py-3 bg-red-500 text-white rounded-lg flex items-center gap-2 mx-auto hover:bg-red-600 transition-colors">
                      <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M7 4a3 3 0 016 0v4a3 3 0 11-6 0V4zm4 10.93A7.001 7.001 0 0017 8a1 1 0 10-2 0A5 5 0 015 8a1 1 0 00-2 0 7.001 7.001 0 006 6.93V17H6a1 1 0 100 2h8a1 1 0 100-2h-3v-2.07z" clipRule="evenodd" /></svg> Start Recording
                    </button>
                  ) : (
                    <div className="space-y-3">
                      <div className="flex items-center justify-center gap-2 text-red-500">
                        <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
                        Recording: {formatDuration(recordingDuration)}
                      </div>
                      <button onClick={stopRecording} className="px-6 py-3 bg-gray-600 text-white rounded-lg flex items-center gap-2 mx-auto hover:bg-gray-700 transition-colors">
                        <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8 7a1 1 0 00-1 1v4a1 1 0 001 1h4a1 1 0 001-1V8a1 1 0 00-1-1H8z" clipRule="evenodd" /></svg> Stop Recording
                      </button>
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                  <div className="flex items-center gap-2 text-green-700 mb-2">
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" /></svg>
                    <span className="font-medium">{audioFileName ? 'Audio File Uploaded' : 'Recording Complete'}</span>
                  </div>
                  {audioFileName && <p className="text-sm text-gray-600 truncate">File: {audioFileName}</p>}
                  {recordingDuration > 0 && <p className="text-sm text-gray-600">Duration: {formatDuration(recordingDuration)}</p>}
                  {audioUrl && <audio controls src={audioUrl} className="w-full mt-3" />}
                </div>
                <div className="flex gap-2 justify-center">
                  <button onClick={clearAudio} className="px-4 py-2 bg-red-500 text-white rounded-lg flex items-center gap-2 hover:bg-red-600 transition-colors">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg> Remove
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* PREDICT BUTTON */}
        {hasInputs && (
          <div className="text-center mb-8">
            <button onClick={predictAge} disabled={isProcessing} className="px-8 py-4 bg-gradient-to-r from-purple-600 to-indigo-600 text-white rounded-xl font-semibold text-lg flex items-center gap-2 mx-auto hover:from-purple-700 hover:to-indigo-700 transition-all transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none">
              {isProcessing ? (
                <><svg className="w-6 h-6 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" /></svg>Processing...</>
              ) : (
                <><svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" /></svg>Predict Age</>
              )}
            </button>
          </div>
        )}

        {/* ERROR DISPLAY */}
        {error && (
          <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mb-6" role="alert">
            <p className="font-bold">Error</p>
            <p>{error}</p>
          </div>
        )}

        {/* RESULTS SECTION */}
        {(predictions.image || predictions.audio || predictions.combined !== null) && (
          <div className="bg-white shadow-lg rounded-xl p-6">
            <h2 className="text-2xl font-semibold mb-6 text-center text-indigo-700">Prediction Results</h2>
            <div className="space-y-6">
              <div className="grid md:grid-cols-2 gap-4">
                {predictions.image && (
                  <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                    <h3 className="font-semibold text-blue-700 mb-2 flex items-center gap-2">
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" /></svg>
                      Image Prediction
                    </h3>
                    {predictions.image.status === 'success' ? <p className="text-2xl font-bold text-blue-600">{predictions.image.age} years</p> : <p className="text-red-500">Failed: {predictions.image.error}</p>}
                  </div>
                )}
                {predictions.audio && (
                  <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                    <h3 className="font-semibold text-green-700 mb-2 flex items-center gap-2">
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" /></svg>
                      Audio Prediction
                    </h3>
                    {predictions.audio.status === 'success' ? (
                      <div>
                        <p className="text-2xl font-bold text-green-600 mb-2">{predictions.audio.age_years} years</p>
                        {predictions.audio.gender_probs && <div className="text-sm text-gray-600"><p className="font-medium mb-1">Gender Probabilities:</p>{Object.entries(predictions.audio.gender_probs).map(([gender, prob]) => (<div key={gender} className="flex justify-between"><span className="capitalize">{gender}:</span><span>{(prob * 100).toFixed(1)}%</span></div>))}</div>}
                      </div>
                    ) : <p className="text-red-500">Failed: {predictions.audio.error}</p>}
                  </div>
                )}
              </div>
              {predictions.combined !== null && predictions.combined !== undefined && (
                <div className="bg-gradient-to-r from-purple-50 to-indigo-50 border border-purple-200 rounded-lg p-6 text-center">
                  <h3 className="font-semibold text-purple-700 mb-2 flex items-center justify-center gap-2 text-lg">
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" /></svg>
                    Combined Age Prediction
                  </h3>
                  <p className="text-4xl font-bold text-purple-600 mb-2">{predictions.combined} years</p>
                  <p className="text-sm text-gray-600">Average of {predictions.image && predictions.audio ? 'image and audio' : predictions.image ? 'image only' : 'audio only'} predictions</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* INSTRUCTIONS */}
        {!hasInputs && !isProcessing && (
          <div className="bg-white shadow-lg rounded-xl p-6 text-center mt-8">
            <h3 className="text-lg font-semibold mb-4 text-gray-700">How to Use</h3>
            <div className="text-gray-600 space-y-2">
              <p>1. Upload an image or use your webcam to capture a photo.</p>
              <p>2. Upload an audio file or record your voice.</p>
              <p>3. Click "Predict Age" to get an AI-powered age estimation.</p>
              <p className="text-sm text-gray-500 mt-4">You can use either input alone, or both together for a combined prediction.</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
