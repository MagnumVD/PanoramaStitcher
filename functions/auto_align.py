import numpy as np
import cv2
from PyQt6.QtWidgets import QApplication


def _normalize_exposure(gray):
    # apply CLAHE for local contrast normalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    # global stretch to [0, 255]
    norm = cv2.normalize(equalized, None, 0, 255, cv2.NORM_MINMAX)
    return norm


def auto_align(main_window):
    if len(main_window.images) < 2:
        message = "Need at least 2 images to auto-align."
        print(message)
        main_window.status.showMessage(message, 4000)
        return

    main_window.status.showMessage("Detecting features...")
    QApplication.processEvents()

    # prefer SIFT, fall back to ORB
    try:
        detector = cv2.SIFT_create(nfeatures=4000)
        norm = cv2.NORM_L2
        use_knn = True
    except AttributeError:
        detector = cv2.ORB_create(nfeatures=4000)
        norm = cv2.NORM_HAMMING
        use_knn = False

    # precompute keypoints and descriptors for all images
    features = []
    for entry in main_window.images:
        gray = main_window._load_gray_display(entry)
        gray = _normalize_exposure(gray)
        kp, desc = detector.detectAndCompute(gray, None)
        features.append((kp, desc))

    bf = cv2.BFMatcher(norm, crossCheck=not use_knn)

    # selected image is the reference
    main_window.selected_entry.H = np.eye(3)
    main_window._apply_H_to_item(main_window.selected_entry)
    aligned_entries = [main_window.selected_entry]
    aligned_count = 1

    for i, entry in enumerate(main_window.images[1:], start=1):
        kp_src, desc_src = features[i]

        if desc_src is None or len(kp_src) < 4:
            message = f"Skipped {entry.name}: no features detected."
            print(message)
            main_window.status.showMessage(message, 3000)
            continue

        # find the aligned image with the most inliers
        best_matches = []
        best_ref_idx = None
        best_H_feat = None
        for j, ref_entry in enumerate(aligned_entries):
            kp_ref, desc_ref = features[main_window.images.index(ref_entry)]

            if use_knn:
                raw = bf.knnMatch(desc_src, desc_ref, k=2)
                # slightly relaxed ratio to handle residual exposure differences
                good = [m for m, n in raw if m.distance < 0.80 * n.distance]
            else:
                raw = bf.match(desc_src, desc_ref)
                good = sorted(raw, key=lambda x: x.distance)[:200]

            if len(good) >= 4 and len(good) > len(best_matches):
                src_pts = np.float32([kp_src[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                H_feat, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if H_feat is not None:
                    best_matches = good
                    best_ref_idx = j
                    best_H_feat = H_feat

        if best_H_feat is None:
            message = f"Skipped {entry.name}: could not align to any reference."
            print(message)
            main_window.status.showMessage(message, 3000)
            continue

        # compose with the reference's homography
        ref_H = aligned_entries[best_ref_idx].H
        entry.H = ref_H @ best_H_feat
        main_window._apply_H_to_item(entry)

        if main_window.selected_entry is entry:
            main_window._sync_spinboxes_from_H(entry.H)

        aligned_entries.append(entry)
        aligned_count += 1
        message = f"Aligned {entry.name}: {len(best_matches)} matches used"
        print(message)
        main_window.status.showMessage(message, 3000)
        QApplication.processEvents()

    main_window.status.showMessage(
        f"Auto-align done: {aligned_count}/{len(main_window.images)} images aligned.", 6000
    )