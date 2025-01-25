import argparse
from datetime import datetime
import logging
import os
import threading
import time
import cv2
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    BitsAndBytesConfig,
)
import sys


def setup_logging(log_to_file, log_file_path="alerts.log"):
    """
    Sets up the logging configuration for the entire application.
    If log_to_file is True, messages will also be written to a specified file.
    """
    handlers = [logging.StreamHandler()]
    if log_to_file:
        handlers.append(logging.FileHandler(log_file_path))
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=handlers)


class RecordingManager:
    """
    Class that manages video recording. Can record continuously or by triggers.
    """

    def __init__(self, record_mode=None):
        """
        Initializes the recording manager with a specified mode.

        :param record_mode: The mode for starting/stopping recording:
            None - no recording,
            "od" - left as a placeholder (used to be object detection),
            "infy"/"infn" - for yes/no triggers.
        """
        self.record_mode = record_mode
        self.recording = False
        self.video_writer = None
        self.video_out_path = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
        self.last_detection_time = time.time()

    def start_recording(self, frame):
        """
        Starts video recording given an initial frame (to set up dimensions, codec, etc.).
        """
        if not self.recording and self.record_mode:
            height, width, _ = frame.shape
            self.video_writer = cv2.VideoWriter(
                self.video_out_path,
                cv2.VideoWriter_fourcc(*"XVID"),
                20.0,
                (width, height),
            )
            self.recording = True
            logging.info(f"Started recording video: {self.video_out_path}")

    def stop_recording(self):
        """Stops video recording and releases the VideoWriter resource."""
        if self.recording:
            self.video_writer.release()
            self.recording = False
            logging.info(f"Stopped recording video: {self.video_out_path}")

    def write_frame(self, frame):
        """Writes a single frame to the open video file if currently recording."""
        if self.recording and self.video_writer:
            self.video_writer.write(frame)

    def handle_recording_by_inference(self, inference_result, frame):
        """
        Starts or stops recording based on a yes/no inference result.
        For example, if record_mode == "infy" and the result is "yes", start recording.
        """
        if not self.record_mode:
            return

        if self.record_mode == "infy" and inference_result == "yes":
            self.start_recording(frame)
        elif self.record_mode == "infy" and inference_result == "no":
            self.stop_recording()
        elif self.record_mode == "infn" and inference_result == "no":
            self.start_recording(frame)
        elif self.record_mode == "infn" and inference_result == "yes":
            self.stop_recording()


class ImageUtils:
    """
    Utility class for image-related operations such as saving screenshots.
    """

    @staticmethod
    def save_screenshot(frame):
        """
        Saves a screenshot of the current frame with a timestamped filename.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            cv2.imwrite(filename, frame)
            logging.info(f"Screenshot saved: {filename}")
            print(f"[{timestamp}] Screenshot saved: {filename}")
        except cv2.error as e:
            logging.error(f"OpenCV error saving screenshot: {e}")
            print(f"[{datetime.now().strftime('%Y%m%d_%H%M%S')}] Error saving screenshot: {e}")
        except Exception as e:
            logging.error(f"Error saving screenshot: {e}")
            print(f"[{datetime.now().strftime('%Y%m%d_%H%M%S')}] Error saving screenshot: {e}")


class AlertLogger:
    """
    A simple class to log alerts both to a dedicated file (alerts.log) and to the console.
    """

    @staticmethod
    def log_alert(message):
        """
        Appends an alert message to a log file with a timestamp, and also prints to console.
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            with open("alerts.log", "a") as log_file:
                log_file.write(f"{timestamp} - {message}\n")
            logging.info(f"{timestamp} - {message}")
            print(f"[{timestamp}] Log entry written: {message}")
        except IOError as e:
            logging.error(f"IO error logging alert: {e}")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}] IO error logging alert: {e}")
        except Exception as e:
            logging.error(f"Error logging alert: {e}")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}] Error logging alert: {e}")


class ModelManager:
    """
    Class responsible for loading and managing a Hugging Face Vision2Seq model and processor,
    with optional quantization settings.
    """

    def __init__(self, device, quantization=None):
        """
        Initialize the ModelManager with a torch device and an optional quantization setting.
        """
        self.device = device
        self.model = None
        self.processor = None
        self.quantization = quantization

    def load_local_model(self, model_path):
        """
        Loads a local model from the specified directory path. Optionally applies quantization.
        """
        if not os.path.exists(model_path):
            logging.error(f"Model path {os.path.abspath(model_path)} does not exist.")
            return False
        if not os.path.isdir(model_path):
            logging.error(f"Model path {os.path.abspath(model_path)} is not a directory.")
            return False

        try:
            logging.info(f"Attempting to load model from {os.path.abspath(model_path)}")
            quant_config = self._get_quant_config()

            self.model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                quantization_config=quant_config,
            ).eval()

            if not self.quantization:
                self.model.to(self.device)
                if torch.cuda.is_available():
                    # half precision if no explicit quantization
                    self.model = self.model.half()
                    logging.info("Using FP16 precision for the model.")

            self.processor = AutoProcessor.from_pretrained(model_path)
            logging.info(f"Model loaded successfully from {os.path.abspath(model_path)}")
            return True
        except (OSError, ValueError, ModuleNotFoundError) as e:
            logging.error(f"Error initializing model: {e}")
        except Exception as e:
            logging.error(f"Unexpected error initializing model: {e}")
        return False

    def download_and_load_model(self, repo_id="HuggingFaceTB/SmolVLM-500M-Instruct"):
        """
        Downloads a model from the Hugging Face Hub using its repository ID, then loads it locally.
        """
        try:
            local_model_dir = "model"
            snapshot_download(repo_id=repo_id, local_dir=local_model_dir)
            if not os.path.exists(local_model_dir):
                logging.error(f"Model download failed, directory {os.path.abspath(local_model_dir)} does not exist.")
                return False
            if not os.path.isdir(local_model_dir):
                logging.error(f"Model download failed, path {os.path.abspath(local_model_dir)} is not a directory.")
                return False
            logging.info(f"Model downloaded and initialized at {os.path.abspath(local_model_dir)}")
            return self.load_local_model(local_model_dir)
        except OSError as e:
            logging.error(f"OS error during model download: {e}")
        except Exception as e:
            logging.error(f"Error downloading model: {e}")
        return False

    def _get_quant_config(self):
        if self.quantization == "4bit":
            logging.info("Using 4-bit quantization.")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        return None


class SmolFLO:
    """
    A simplified version of YO-FLO that uses SmolVLM (Vision2Seq model) instead of Florence-2.
    No object detection or bounding box logic, just image-based Q&A / captioning / yes-no inferences.
    """

    def __init__(
        self,
        model_path=None,
        display_inference_rate=False,
        pretty_print=False,
        inference_limit=None,
        webcam_indices=None,
        rtsp_urls=None,
        record=None,
        quantization=None,
    ):
        """
        Initializes the SmolFLO system with various configuration options.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.inference_start_time = None
        self.inference_count = 0
        self.screenshot_active = False
        self.log_to_file_active = False
        self.headless = True

        self.display_inference_rate = display_inference_rate
        self.stop_webcam_flag = threading.Event()
        self.webcam_threads = []
        self.pretty_print = pretty_print
        self.inference_limit = inference_limit
        self.last_inference_time = 0
        self.inference_phrases = []
        self.webcam_indices = webcam_indices if webcam_indices else [0]
        self.rtsp_urls = rtsp_urls if rtsp_urls else []
        self.quantization = quantization
        self.record = record
        self.phrase = None

        self.recording_manager = RecordingManager(record)
        self.model_manager = ModelManager(self.device, self.quantization)

        if model_path:
            self.model_manager.load_local_model(model_path)

    @property
    def model(self):
        return self.model_manager.model

    @property
    def processor(self):
        return self.model_manager.processor

    def update_inference_rate(self):
        try:
            if self.inference_start_time is None:
                self.inference_start_time = time.time()
            else:
                elapsed_time = time.time() - self.inference_start_time
                if elapsed_time > 0:
                    inferences_per_second = self.inference_count / elapsed_time
                    if self.display_inference_rate:
                        logging.info(f"IPS: {inferences_per_second:.2f}")
        except Exception as e:
            logging.error(f"Error updating inference rate: {e}")

    def run_smolvlm_inference(self, image, user_prompt):
        """
        A generic method to query SmolVLM with an image + a text question/command.
        Returns the raw text the model outputs.

        FIX #1 IMPLEMENTED: We now explicitly cast inputs to the model's FP16 dtype.
        """
        try:
            if not self.model or not self.processor:
                logging.error("Model or processor not initialized.")
                return None

            # Build a chat-style message.
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ]
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

            # Preprocessing, but do NOT just .to(device) at full float32
            inputs = self.processor(
                text=prompt,
                images=[image],
                return_tensors="pt",
            )

            # Explicitly cast the inputs to the same dtype as the model (FP16) to avoid overhead
            dtype = next(self.model.parameters()).dtype
            for k, v in inputs.items():
                if torch.is_floating_point(v):
                    inputs[k] = v.to(self.device, dtype=dtype)
                else:
                    inputs[k] = v.to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=30,
                    early_stopping=True,
                    do_sample=False,
                    num_beams=1,
                )
            generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            return generated_texts[0] if generated_texts else None
        except Exception as e:
            logging.error(f"Error during SmolVLM inference: {e}")
        return None

    def evaluate_inference_chain(self, image):
        """
        If multiple yes/no questions (phrases) are given,
        ask them one by one and interpret 'yes' in each answer.
        Return "PASS" if majority of answers are "yes" (or some threshold).
        """
        try:
            if not self.inference_phrases:
                logging.error("No inference phrases set.")
                return "FAIL", []

            results = []
            for phrase in self.inference_phrases:
                answer = self.run_smolvlm_inference(image, phrase)
                if not answer:
                    results.append(False)
                    continue
                results.append("yes" in answer.lower())

            positive_count = sum(results)
            total_count = len(results)
            if positive_count > (total_count // 2):
                overall_result = "PASS"
            else:
                overall_result = "FAIL"

            return overall_result, results
        except Exception as e:
            logging.error(f"Error evaluating inference chain: {e}")
            return "FAIL", []

    def set_inference_phrases(self, phrases):
        self.inference_phrases = phrases
        logging.info(f"Inference phrases set: {self.inference_phrases}")

    def download_model(self):
        return self.model_manager.download_and_load_model()

    def start_webcam_detection(self):
        """
        Each webcam thread captures frames, runs single-phrase or chain inferences,
        and displays/logs the result.
        """
        try:
            if self.webcam_threads:
                logging.warning("Webcam detection is already running.")
                return
            self.stop_webcam_flag.clear()

            sources = self.rtsp_urls if self.rtsp_urls else self.webcam_indices
            for source in sources:
                thread = threading.Thread(
                    target=self._webcam_detection_thread, args=(source,)
                )
                thread.start()
                self.webcam_threads.append(thread)
        except Exception as e:
            logging.error(f"Error starting webcam detection: {e}")

    def stop_webcam_detection(self):
        try:
            self.stop_webcam_flag.set()
            for thread in self.webcam_threads:
                thread.join()
            self.webcam_threads = []
            logging.info("Webcam detection stopped")
            if self.recording_manager.recording:
                self.recording_manager.stop_recording()
        except Exception as e:
            logging.error(f"Error stopping webcam detection: {e}")

    def _webcam_detection_thread(self, source):
        try:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                logging.error(f"Could not open video source {source}.")
                return

            window_name = f"SmolVLM Inference Source {source}"
            while not self.stop_webcam_flag.is_set():
                ret, frame = cap.read()
                if not ret:
                    logging.error(f"Failed to capture image from source {source}.")
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error: Failed to capture image from source {source}.")
                    break

                # Convert to PIL for the model
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(image_rgb)

                current_time = time.time()
                if self.inference_limit:
                    time_since_last_inference = current_time - self.last_inference_time
                    if time_since_last_inference < 1 / self.inference_limit:
                        time.sleep(1 / self.inference_limit - time_since_last_inference)
                    current_time = time.time()

                # Single phrase or chain of phrases
                if self.phrase:
                    result = self.run_smolvlm_inference(image_pil, self.phrase)
                    if result:
                        clean_result = result.strip()
                        if self.pretty_print:
                            self._pretty_print_expression(clean_result)
                        else:
                            logging.info(f"Single-phrase result: {clean_result}")
                            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Single-phrase result: {clean_result}")

                        # Check for yes/no triggers
                        if "yes" in clean_result.lower():
                            if self.log_to_file_active:
                                AlertLogger.log_alert("Expression: yes")
                            self.recording_manager.handle_recording_by_inference("yes", frame)
                        elif "no" in clean_result.lower():
                            if self.log_to_file_active:
                                AlertLogger.log_alert("Expression: no")
                            self.recording_manager.handle_recording_by_inference("no", frame)

                        self.inference_count += 1
                        self.update_inference_rate()

                if self.inference_phrases:
                    # Evaluate chain
                    inference_result, phrase_results = self.evaluate_inference_chain(image_pil)
                    logging.info(f"Inference Chain result: {inference_result}, Details: {phrase_results}")
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Inference Chain result: {inference_result}, Details: {phrase_results}")

                    if self.pretty_print:
                        for idx, res in enumerate(phrase_results):
                            print(f"Inference {idx+1}: {'YES' if res else 'NO'}")

                    self.inference_count += 1
                    self.update_inference_rate()

                if self.screenshot_active:
                    # If you want a screenshot for each frame or on some trigger...
                    pass

                # If recording
                if self.recording_manager.recording:
                    self.recording_manager.write_frame(frame)

                # Show frame if not headless
                if not self.headless:
                    cv2.imshow(window_name, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                self.last_inference_time = current_time

            cap.release()
            if not self.headless:
                cv2.destroyWindow(window_name)
            if self.recording_manager.recording:
                self.recording_manager.stop_recording()

        except cv2.error as e:
            logging.error(f"OpenCV error in detection thread {source}: {e}")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] OpenCV error in detection thread {source}: {e}")
        except ModuleNotFoundError as e:
            logging.error(f"ModuleNotFoundError in detection thread {source}: {e}")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ModuleNotFoundError in detection thread {source}: {e}")
        except Exception as e:
            logging.error(f"Error in detection thread {source}: {e}")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error in detection thread {source}: {e}")

    def _pretty_print_expression(self, clean_result):
        """
        Prints expression comprehension results in a nicely formatted block.
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logging.info("\n" + "=" * 50)
            logging.info(f"SmolVLM Output: {clean_result} at {timestamp}")
            logging.info("=" * 50 + "\n")
        except Exception as e:
            logging.error(f"Error in _pretty_print_expression: {e}")


def main():
    """
    Main function to parse command-line arguments, configure and run the SmolFLO system.
    """
    parser = argparse.ArgumentParser(
        description="SmolFLO: A simplified, SmolVLM-based script (no object detection)."
    )
    parser.add_argument("-ph", type=str, help="Yes/No question or any text prompt for the VQA.")
    parser.add_argument("-hl", action="store_true", help="Run in headless mode (no video display).")
    parser.add_argument("-ss", action="store_true", help="Enable screenshot on detection/inference.")
    parser.add_argument("-lf", action="store_true", help="Enable logging alerts to file.")
    parser.add_argument("-ir", action="store_true", help="Display inference rate.")
    parser.add_argument("-pp", action="store_true", help="Enable pretty print for text outputs.")
    parser.add_argument("-il", type=float, help="Limit the inference rate (inferences per second).")
    parser.add_argument("-ic", nargs="+", help="Enable inference chain with specified phrases.")
    parser.add_argument("-wi", nargs="+", type=int, help="Indices of webcams to use.")
    parser.add_argument("-rtsp", nargs="+", type=str, help="RTSP URLs for video streams.")
    parser.add_argument("-r", choices=["od", "infy", "infn"], help="Video recording mode (od/infy/infn).")
    parser.add_argument("-4bit", action="store_true", help="Enable 4-bit quantization.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-mp", type=str, help="Path to the local pre-trained SmolVLM directory.")
    group.add_argument("-dm", action="store_true", help="Download the SmolVLM model from Hugging Face.")

    args = parser.parse_args()
    quantization_mode = "4bit" if getattr(args, '4bit', False) else None

    try:
        setup_logging(args.lf)
        webcam_indices = args.wi if args.wi else [0]
        rtsp_urls = args.rtsp if args.rtsp else []

        # Initialize
        smol_flo = SmolFLO(
            display_inference_rate=args.ir,
            pretty_print=args.pp,
            inference_limit=args.il,
            webcam_indices=webcam_indices,
            rtsp_urls=rtsp_urls,
            record=args.r,
            quantization=quantization_mode,
        )

        if args.dm:
            # Download model from HF
            if not smol_flo.download_model():
                return
        else:
            # Load from local path
            if not os.path.exists(args.mp):
                logging.error(f"Model path {args.mp} does not exist.")
                return
            if not os.path.isdir(args.mp):
                logging.error(f"Model path {args.mp} is not a directory.")
                return
            smol_flo.model_manager.load_local_model(args.mp)

        # Optional prompts and chain phrases
        if args.ph:
            smol_flo.phrase = args.ph
        if args.ic:
            smol_flo.set_inference_phrases(args.ic)

        smol_flo.headless = args.hl
        smol_flo.screenshot_active = args.ss
        smol_flo.log_to_file_active = args.lf

        # Start webcam or RTSP threads
        smol_flo.start_webcam_detection()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            smol_flo.stop_webcam_detection()
        finally:
            smol_flo.stop_webcam_detection()

    except Exception as e:
        logging.error(f"An error occurred during main loop: {e}")
    else:
        input("Press Enter to stop...")
        smol_flo.stop_webcam_detection()


if __name__ == "__main__":
    main()
