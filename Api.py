from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import StreamingResponse
import cv2

app = FastAPI()
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace;boundary=frame")

@app.post("/start_stream")
async def start_stream(background_tasks: BackgroundTasks):
    global camera
    camera = cv2.VideoCapture(0)

@app.post("/stop_stream")
async def stop_stream():
    global camera
    camera.release()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
