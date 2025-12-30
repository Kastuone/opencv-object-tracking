import cv2
import numpy as np

def detect_specific_target_circle():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Hata: Kamera başlatılamadı!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 70)
    cap.set(cv2.CAP_PROP_CONTRAST, 70)

    # Kamera merkez koordinatları
    frame_center_x = 1280 // 2  # 640
    frame_center_y = 720 // 2   # 360

    # Hedef dairenin RGBA(66,65,63,255) rengine göre HSV aralığı
    # Bu değerleri canlı olarak, hedef tahtanızı kameraya tutarak ayarlamalısınız!
    lower_hsv_gray = np.array([0, 0, 0])
    upper_hsv_gray = np.array([180, 60, 100])

    params = {
        'dp': 1.5,        # Hough çözünürlük oranı (daha yüksek hassasiyet)
        'param1': 50,     # Canny kenar eşiği
        'param2': 50,     # Akümülatör eşiği (düşük = hassas)
        'minDist': 30,    # Minimum merkez mesafesi
        'minRadius': 2,   # Küçük daire algılaması için
        'maxRadius': 15   # Büyük daireleri dışlamak için
    }

    # Son tespit edilen daire bilgilerini saklamak için değişken
    last_detected_circle = None # (x, y, r) formatında saklayacak

    # Takip süresi için yeni değişkenler
    frames_since_last_detection = 0
    max_frames_to_hold = 10 # Daire kaybolduktan sonra kaç kare daha son pozisyonu tutsun (yaklaşık 0.1 sn)

    print("\n--- Hedef Daire Algılama Ayarları (Klavye Tuşları) ---")
    print(" 'w'/'s' : param2 (kesinlik)")
    print(" 'a'/'d' : minDist")
    print(" 'r'/'f' : minRadius")
    print(" 't'/'g' : maxRadius")
    print(" 'z'/'x' : HSV V üst sınırı")
    print(" 'c'/'v' : HSV S üst sınırı")
    print(" 'm'/'n' : Max Frames To Hold (Kaybolma toleransı)")
    print(" 'q' : çıkış\n")
    print(f"Kamera merkezi: ({frame_center_x}, {frame_center_y}) -> (0, 0) olarak ayarlandı\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Kare alınamadı!")
            break

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, lower_hsv_gray, upper_hsv_gray)

        # Gürültü azaltma
        mask = cv2.GaussianBlur(mask, (9, 9), 2)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, params['dp'],
                                   params['minDist'], param1=params['param1'],
                                   param2=params['param2'],
                                   minRadius=params['minRadius'],
                                   maxRadius=params['maxRadius'])

        current_detection_found_this_frame = False
        if circles is not None:
            circles = np.round(circles[:, 0, :]).astype("int")
            if len(circles) > 0:
                # Birden fazla daire algılanırsa yarıçapı en büyük olanı seç
                largest = max(circles, key=lambda c: c[-1])

                # Eğer geçerli bir daire algılandıysa
                if largest is not None:
                    last_detected_circle = largest # Son bilinen konumu güncelle
                    frames_since_last_detection = 0 # Algılama sayacını sıfırla
                    current_detection_found_this_frame = True # Bu karede tespit edildi bayrağı

        # Çizim ve bilgi gösterme:
        # Eğer bir daire algılandıysa VEYA son algılama eşik süresi içinde ise
        detected_coords = None
        if last_detected_circle is not None and frames_since_last_detection < max_frames_to_hold:
            x, y, r = last_detected_circle
            
            # Merkez koordinatlarını kamera merkezine göre hesapla (0,0 = kamera merkezi)
            center_relative_x = x - frame_center_x
            center_relative_y = frame_center_y - y  # Y eksenini ters çevir (üst = pozitif)
            
            cv2.circle(frame, (x, y), r, (0, 255, 0), 1) # Yeşil daire (kalınlık 1 olarak ayarlandı)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1) # Kırmızı merkez - Yarıçapı 2'ye düşürüldü
            cv2.putText(frame, f"Merkez: ({center_relative_x},{center_relative_y})", (x+10, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            detected_coords = (center_relative_x, center_relative_y) # Merkez koordinatları

            # Konsol çıktısı sadece gerçekten yeni bir algılama olduğunda veya kaybolduğunda
            if current_detection_found_this_frame:
                print(f"Hedef Daire Tespit Edildi: Merkez=({center_relative_x}, {center_relative_y})")
            else:
                print(f"Daire geçici olarak kayıp. Son bilinen konum kullanılıyor ({max_frames_to_hold - frames_since_last_detection} kare kaldı).")

        else: # Daire bulunamadıysa ve eşik aşıldıysa
            last_detected_circle = None # Konumu sıfırla
            print("Daire algılanamıyor ve takip süresi aşıldı.")

        # Eğer bu karede bir daire algılanmadıysa, sayacı artır
        if not current_detection_found_this_frame:
            frames_since_last_detection += 1
            if frames_since_last_detection >= max_frames_to_hold:
                # Eşik aşıldığında mesajı sadece bir kez yazdır
                if frames_since_last_detection == max_frames_to_hold:
                    print("Hedef tamamen kayıp olarak işaretlendi.")

        # Kamera merkezini gösteren mavi nokta ve çizgiler
        cv2.circle(frame, (frame_center_x, frame_center_y), 3, (255, 0, 0), -1)  # Mavi nokta
        cv2.line(frame, (frame_center_x-10, frame_center_y), (frame_center_x+10, frame_center_y), (255, 0, 0), 1)
        cv2.line(frame, (frame_center_x, frame_center_y-10), (frame_center_x, frame_center_y+10), (255, 0, 0), 1)

        # Bilgiler
        cv2.putText(frame, f"P2:{params['param2']} MinD:{params['minDist']}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"MinR:{params['minRadius']} MaxR:{params['maxRadius']}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"HSV V:{upper_hsv_gray[:][2]} S:{upper_hsv_gray[:][1]}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Hold Frames: {max_frames_to_hold}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        # Kameranın tam ortasındaki koordinat bilgisini gösteren satır kaldırıldı
        # cv2.putText(frame, f"Merkez (0,0): ({frame_center_x},{frame_center_y})", (10, 150),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        if detected_coords:
            cv2.putText(frame, f"Hedef Bulundu: X={detected_coords[:][0]}, Y={detected_coords[:][1]}", (10, frame.shape[:][0]-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Hedef Yok", (10, frame.shape[:][0]-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Kamera Görüntüsü", frame)
        cv2.imshow("Renk Maskesi", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('w'):
            params['param2'] = min(200, params['param2'] + 1)
            print(f"param2 artirildi: {params['param2']}")
        elif key == ord('s'):
            params['param2'] = max(1, params['param2'] - 1)
            print(f"param2 azaltildi: {params['param2']}")
        elif key == ord('a'):
            params['minDist'] = max(1, params['minDist'] - 1)
            print(f"minDist azaltildi: {params['minDist']}")
        elif key == ord('d'):
            params['minDist'] += 1
            print(f"minDist artirildi: {params['minDist']}")
        elif key == ord('r'):
            params['minRadius'] = max(1, params['minRadius'] - 1)
            print(f"minRadius azaltildi: {params['minRadius']}")
        elif key == ord('f'):
            params['minRadius'] += 1
            print(f"minRadius artirildi: {params['minRadius']}")
        elif key == ord('t'):
            params['maxRadius'] = max(1, params['maxRadius'] - 1)
            print(f"maxRadius azaltildi: {params['maxRadius']}")
        elif key == ord('g'):
            params['maxRadius'] += 1
            print(f"maxRadius artirildi: {params['maxRadius']}")
        elif key == ord('z'):
            upper_hsv_gray[:][2] = min(255, upper_hsv_gray[:][2] + 1)
            print(f"HSV Upper Value artirildi: {upper_hsv_gray[:][2]}")
        elif key == ord('x'):
            upper_hsv_gray[:][2] = max(0, upper_hsv_gray[:][2] - 1)
            print(f"HSV Upper Value azaltildi: {upper_hsv_gray[:][2]}")
        elif key == ord('c'):
            upper_hsv_gray[:][1] = min(255, upper_hsv_gray[:][1] + 1)
            print(f"HSV Upper Saturation artirildi: {upper_hsv_gray[:][1]}")
        elif key == ord('v'):
            upper_hsv_gray[:][1] = max(0, upper_hsv_gray[:][1] - 1)
            print(f"HSV Upper Saturation azaltildi: {upper_hsv_gray[:][1]}")
        elif key == ord('m'):
            max_frames_to_hold = min(100, max_frames_to_hold + 1)
            print(f"Max Frames to Hold artirildi: {max_frames_to_hold}")
        elif key == ord('n'):
            max_frames_to_hold = max(1, max_frames_to_hold - 1)
            print(f"Max Frames to Hold azaltildi: {max_frames_to_hold}")

    cap.release()
    cv2.destroyAllWindows()

if _name_ == "_main_":
    detect_specific_target_circle()