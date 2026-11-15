
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from ultralytics import YOLO
from collections import Counter

# Loading the trained model
model = YOLO(r"E:\hotel_facility_detection\hotel_best.pt")

st.title("Hotel Facility Detection")
#st.write("Object detection and counting using YOLO11")

# Image selection
st.subheader("Select an Image")

option = st.radio(
    "Choose image source",
    ["Camera", "Upload Image"],
    horizontal=True
)

image = None

if option == "Camera":

    image_file = st.camera_input("Capture hotel room image")

else:

    image_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

# Reading the image
if image_file is not None:

    image_bytes = np.frombuffer(
        image_file.getvalue(),
        np.uint8
    )

    image = cv2.imdecode(
        image_bytes,
        cv2.IMREAD_COLOR
    )

    
# Display original image
    st.subheader("Original Image")

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.axis("off")

    st.pyplot(fig)
    plt.close(fig)
   
    

    # Detect button
    if st.button("Detect Objects", type="primary"):

        results = model.predict(
            source=image,
            conf=0.38,
            imgsz=640,
            verbose=False
        )

        result = results[0]

        st.success("Detection completed!")

        # Display predicted image
        st.subheader("3. Predicted Image")

        predicted_image = result.plot()

        fig, ax = plt.subplots(figsize=(6, 6))

        ax.imshow(
            cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGB)
        )

        ax.axis("off")

        st.pyplot(fig)
        plt.close(fig)

        # Getting predicted class IDs
        if result.boxes is not None and len(result.boxes) > 0:

            class_ids = (
                result.boxes.cls.cpu().numpy().astype(int)
            )

            counts = Counter(class_ids)

        else:

            counts = Counter()

        # Counting all classes
        class_names = result.names

        class_counts = {}

        for class_id, class_name in class_names.items():

            class_counts[class_name] = counts.get(class_id, 0)

        # Creating DataFrame
        df = pd.DataFrame(
            list(class_counts.items()),
            columns=["Facility", "Count"]
        )

        # Summary
        total_objects = sum(class_counts.values())

        detected_classes = sum(
            1 for count in class_counts.values() if count > 0
        )

        st.subheader("Detection Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Objects", total_objects)

        col2.metric("Classes Detected", detected_classes)

        col3.metric(
            "Classes Not Detected",
            len(class_counts) - detected_classes
        )

        
        # Bar chart
        st.subheader("Object Count Visualization")

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.bar(
            df["Facility"],
            df["Count"]
        )

        ax.set_xlabel("Hotel Facilities")
        ax.set_ylabel("Number of Objects")
        ax.set_title("Class-wise Object Count")

        plt.xticks(rotation=45)
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        # Detected and not detected classes
        st.subheader("Detection Legend")

        detected = df[df["Count"] > 0]
        not_detected = df[df["Count"] == 0]

        col1, col2 = st.columns(2)

        with col1:

            st.write("**Detected Facilities**")

            for _, row in detected.iterrows():

                st.write(
                    f"🟢 {row['Facility']} : {row['Count']}"
                )

        with col2:

            st.write("**Not Detected**")

            for _, row in not_detected.iterrows():

                st.write(f"⚪ {row['Facility']} : 0")

        # Individual detection confidence
        st.subheader("8. Detection Details")

        details = []

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(box.cls.item())

                details.append({
                    "Class": class_names[class_id],
                    "Confidence": round(
                        float(box.conf.item()), 3
                    )
                })

        if details:

            details_df = pd.DataFrame(details)

            st.dataframe(
                details_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.warning("No objects detected.")

        # Download report
        st.subheader("9. Download Report")

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Facility Count CSV",
            data=csv,
            file_name="hotel_facility_count.csv",
            mime="text/csv"
        )

else:

    st.info("Capture or upload an image to start detection.")
