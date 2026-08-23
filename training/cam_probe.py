import cv2
for i in range(4):
    cap = cv2.VideoCapture(i)
    opened = cap.isOpened()
    ok, frame = (False, None)
    if opened:
        ok, frame = cap.read()
    shape = frame.shape if ok and frame is not None else None
    print(f"index {i}: isOpened={opened} read_ok={ok} frame_shape={shape}")
    cap.release()
