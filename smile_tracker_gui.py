import numpy as np
import cv2
from tensorflow.keras.models import load_model
import time
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import os

class SmileTrackerTensorFlowGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Smile Engagement Tracker")
        self.root.geometry("1000x750")
        
      
        self.is_running = False
        self.cap = None
        
      
        self.model = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.IMG_SIZE = 64
        
        
        self.face_detected_time = 0.0
        self.smile_detected_time = 0.0
        self.last_frame_time = None
        self.start_of_code_time = None
        
       
        self.current_prediction = 0.0
        self.is_smiling = False
        self.face_detected = False
        
        
        self.setup_gui()
        
      
        self.load_model()
        
    def setup_gui(self):
        style = ttk.Style()
        style.configure('Title.TLabel', font=('Arial', 28, 'bold'))
        style.configure('Stats.TLabel', font=('Arial', 14))
        style.configure('Status.TLabel', font=('Arial', 12))
        
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        
        title_label = ttk.Label(main_frame, text="Smile Engagement Tracker", 
                               style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=3, pady=10)
        
        subtitle_label = ttk.Label(main_frame, text="Created by: Jake Sacilotto, Seungyeop Shin, Eito Nishikawa", 
                           font=('Helvetica', 12, 'italic'),
                           anchor='center',
                           justify='center')
        subtitle_label.grid(row=1, column=0, columnspan=3, pady=(0, 20))
        
        
        video_frame = ttk.LabelFrame(main_frame, text="Camera Feed", padding="10")
        video_frame.grid(row=2, column=0, columnspan=2, padx=(0, 10), sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.canvas = tk.Canvas(video_frame, width=640, height=480, bg='black')
        self.canvas.pack()
        
        info_frame = ttk.LabelFrame(main_frame, text="Real-time Info", padding="10")
        info_frame.grid(row=2, column=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.face_status_var = tk.StringVar(value="❌ No Face")
        self.smile_status_var = tk.StringVar(value="😐 Not Smiling")
        self.confidence_var = tk.StringVar(value="Confidence: 0%")
        
        ttk.Label(info_frame, textvariable=self.face_status_var,
                 font=('Arial', 16)).pack(pady=10)
        ttk.Label(info_frame, textvariable=self.smile_status_var,
                 font=('Arial', 16)).pack(pady=10)
        ttk.Label(info_frame, textvariable=self.confidence_var,
                 font=('Arial', 14)).pack(pady=10)
        
       
        ttk.Label(info_frame, text="Smile Confidence:", 
                 font=('Arial', 12)).pack(pady=(20, 5))
        self.confidence_meter = ttk.Progressbar(info_frame, length=200, 
                                               mode='determinate', maximum=100)
        self.confidence_meter.pack(pady=5)
        
       
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=3, column=0, columnspan=3, pady=20)
        
        self.start_button = ttk.Button(control_frame, text="▶️ Start Detection", 
                                      command=self.start_detection,
                                      width=20)
        self.start_button.grid(row=0, column=0, padx=10)
        
        self.stop_button = ttk.Button(control_frame, text="⏹️ Stop Detection", 
                                     command=self.stop_detection,
                                     state='disabled',
                                     width=20)
        self.stop_button.grid(row=0, column=1, padx=10)
        
        #self.reset_button = ttk.Button(control_frame, text="🔄 Reset Stats", 
        #                              command=self.reset_stats,
        #                              width=20)
        #self.reset_button.grid(row=0, column=2, padx=10)
        
       
        stats_frame = ttk.LabelFrame(main_frame, text="Session Statistics", padding="15")
        stats_frame.grid(row=4, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))

        self.total_time_var = tk.StringVar(value="Total Time: 0:00")
        self.face_time_var = tk.StringVar(value="Face Time: 0:00")
        self.smile_time_var = tk.StringVar(value="Smile Time: 0:00")
        self.engagement_var = tk.StringVar(value="Engagement Score: 0%")
        ttk.Label(stats_frame, textvariable=self.total_time_var, 
                style='Stats.TLabel', anchor='center').grid(row=0, column=0, pady=5, sticky=tk.EW)
        ttk.Label(stats_frame, textvariable=self.face_time_var,
                style='Stats.TLabel', anchor='center').grid(row=1, column=0, pady=5, sticky=tk.EW)
        ttk.Label(stats_frame, textvariable=self.smile_time_var,
                style='Stats.TLabel', anchor='center').grid(row=2, column=0, pady=5, sticky=tk.EW)
        ttk.Label(stats_frame, textvariable=self.engagement_var,
                style='Stats.TLabel', foreground='green', anchor='center').grid(row=3, column=0, pady=5, sticky=tk.EW)

        ttk.Label(stats_frame, text="Engagement Level:", 
                font=('Arial', 12), anchor='center').grid(row=4, column=0, pady=(10, 5), sticky=tk.EW)
        self.engagement_bar = ttk.Progressbar(stats_frame, length=400, 
                                            mode='determinate', maximum=100)
        self.engagement_bar.grid(row=5, column=0, pady=(0, 5))
        stats_frame.columnconfigure(0, weight=1)

        
        #self.status_var = tk.StringVar(value="Ready - Click Start to begin")
        #status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
        #                      style='Status.TLabel', relief=tk.SUNKEN)
        #status_bar.grid(row=5, column=0, columnspan=3, pady=(10, 0), sticky=(tk.W, tk.E))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
    def load_model(self):
        try:
            if os.path.exists('model.keras'):
                self.model = load_model('model.keras')
                #self.status_var.set("✅ Model loaded successfully")
                print("Model loaded successfully")
            else:
                #self.status_var.set("❌ model.keras not found")
                messagebox.showerror("Error", "model.keras file not found!\nPlease ensure the model file is in the same directory.")
                self.start_button['state'] = 'disabled'
        except Exception as e:
            #self.status_var.set(f"❌ Model loading error: {str(e)}")
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            self.start_button['state'] = 'disabled'
    
    def format_time(self, seconds):
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"
    
    def update_stats_display(self):
        if self.start_of_code_time:
            total_time = time.time() - self.start_of_code_time
            self.total_time_var.set(f"Total Time: {self.format_time(total_time)}")
            self.face_time_var.set(f"Face Time: {self.format_time(self.face_detected_time)}")
            self.smile_time_var.set(f"Smile Time: {self.format_time(self.smile_detected_time)}")
            
            if self.face_detected_time > 0:
                engagement = (self.smile_detected_time / self.face_detected_time) * 100
                self.engagement_var.set(f"Engagement Score: {engagement:.1f}%")
                self.engagement_bar['value'] = engagement
            else:
                self.engagement_var.set("Engagement Score: N/A")
                self.engagement_bar['value'] = 0
    
    def detection_loop(self):
        self.cap = cv2.VideoCapture(0)
        self.last_frame_time = time.time()
        self.start_of_code_time = time.time()
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            current_time = time.time()
            frame_time = current_time - self.last_frame_time
            self.last_frame_time = current_time
            
            display_frame = frame.copy()
            
            if len(faces) > 0:
                self.face_detected = True
                self.face_detected_time += frame_time
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
                    
                    face_roi = gray[y:y+h, x:x+w]
                    
                    try:
                        face_resized = cv2.resize(face_roi, (self.IMG_SIZE, self.IMG_SIZE))
                        face_normalized = face_resized.astype("float32") / 255.0
                        face_input = np.expand_dims(face_normalized, axis=(0, -1))
                        
                        prediction = self.model.predict(face_input, verbose=0)[0][0]
                        self.current_prediction = float(prediction)
                        
                        if prediction > 0.40:
                            self.smile_detected_time += frame_time
                            self.is_smiling = True
                            label = f"Smiling ({prediction:.2f})"
                            color = (0, 255, 0)  
                        else:
                            self.is_smiling = False
                            label = f"Not Smiling ({prediction:.2f})"
                            color = (0, 0, 255)  
                        
                        cv2.putText(display_frame, label, (x, y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                        
                    except Exception as e:
                        print(f"Face preprocessing failed: {e}")
                        self.is_smiling = False
                        self.current_prediction = 0.0
            else:
                self.face_detected = False
                self.is_smiling = False
                self.current_prediction = 0.0
            
            self.update_realtime_info()
            
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image=img)
            
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas.image = photo  
            
            self.root.after(0, self.update_stats_display)
          
            time.sleep(0.03)
        
        self.cap.release()
    
    def update_realtime_info(self):
        if self.face_detected:
            self.face_status_var.set("✅ Face Detected")
            if self.is_smiling:
                self.smile_status_var.set("😊 Smiling!")
                #self.status_var.set("Great smile! Keep it up!")
            else:
                self.smile_status_var.set("😐 Not Smiling")
               # self.status_var.set("Try smiling! 😊")
        else:
            self.face_status_var.set("❌ No Face")
            #self.smile_status_var.set("😐 Not Smiling")
            #self.status_var.set("Position your face in front of the camera")
        
        confidence_percent = self.current_prediction * 100
        self.confidence_var.set(f"Confidence: {confidence_percent:.1f}%")
        self.confidence_meter['value'] = confidence_percent
    
    def start_detection(self):
        if not self.is_running and self.model is not None:
            self.is_running = True
            
            self.start_button['state'] = 'disabled'
            self.stop_button['state'] = 'normal'
            
            self.detection_thread = threading.Thread(target=self.detection_loop)
            self.detection_thread.daemon = True
            self.detection_thread.start()
            
            #self.status_var.set("Detection started! Look at the camera and smile 😊")
    
    def stop_detection(self):
    
        if self.is_running:
            self.is_running = False
            
          
            self.start_button['state'] = 'normal'
            self.stop_button['state'] = 'disabled'
            
           
            self.show_final_stats()
            
        
            self.canvas.delete("all")
            self.canvas.create_text(320, 240, text="Press Start to begin", 
                                   font=('Arial', 24), fill='white')
            
            #self.status_var.set("Detection stopped")
    
    def reset_stats(self):

        self.face_detected_time = 0.0
        self.smile_detected_time = 0.0
        self.update_stats_display()
        #self.status_var.set("Statistics reset")
    
    def show_final_stats(self):
      
        if self.start_of_code_time:
            total_time = time.time() - self.start_of_code_time
            
            print("-" * 66)
            print(f"Total Time Code Ran For: {total_time:.2f} seconds".center(66))
            print(f"Total Time a Face Was Detected: {self.face_detected_time:.2f} seconds".center(66))
            print(f"Total Time a Smile Was Detected: {self.smile_detected_time:.2f} seconds".center(66))
            
            if self.face_detected_time > 0:
                engagement_score = self.smile_detected_time / self.face_detected_time
                print(f"Engagement Score: {engagement_score:.2%}".center(66))
            else:
                print("No face was detected, therefore engagement score doesn't apply".center(66))
            print("-" * 66)
            
            stats_text = f"Session Statistics:\n\n"
            stats_text += f"Total Time: {self.format_time(total_time)}\n"
            stats_text += f"Face Detection Time: {self.format_time(self.face_detected_time)}\n"
            stats_text += f"Smile Detection Time: {self.format_time(self.smile_detected_time)}\n"
            
            if self.face_detected_time > 0:
                engagement = (self.smile_detected_time / self.face_detected_time) * 100
                stats_text += f"\nEngagement Score: {engagement:.1f}%"
            else:
                stats_text += f"\nNo engagement score (no face detected)"
            
            messagebox.showinfo("Session Complete", stats_text)

def main():
    root = tk.Tk()
    app = SmileTrackerTensorFlowGUI(root)
    
    def on_closing():
        if app.is_running:
            app.stop_detection()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()