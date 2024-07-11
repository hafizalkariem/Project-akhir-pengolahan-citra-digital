import streamlit as st
import cv2
from PIL import Image, ImageEnhance, ExifTags
import numpy as np
from sklearn.cluster import KMeans
from geopy.geocoders import Nominatim
import gmplot
import os
import tempfile

face_cascade = cv2.CascadeClassifier('./detector/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./detector/haarcascade_eye.xml')
geolocator = Nominatim(user_agent="streamlit_app")

def detect_faces(our_image):
    new_img = np.array(our_image.convert("RGB"))
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(gray, 1.1, 10)
    for (x, y, w, h) in faces:
        cv2.rectangle(new_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return new_img, faces

def detect_eye(our_image):
    new_img = np.array(our_image.convert("RGB"))
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 25)
    for (x, y, w, h) in eyes:
        cv2.rectangle(new_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return new_img, eyes

def cartoonize_image(our_image):
    new_img = np.array(our_image.convert("RGB"))
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(new_img, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def kmeans_clustering(our_image, k):
    img = np.array(our_image.convert('RGB'))
    img = img / 255.0
    img_flat = img.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(img_flat)
    clustered_img_flat = kmeans.cluster_centers_[kmeans.labels_]
    clustered_img = clustered_img_flat.reshape(img.shape)
    clustered_img = np.clip(clustered_img * 255, 0, 255).astype(np.uint8)
    return clustered_img


def convert_to_degrees(value):
    d = float(value[0])
    m = float(value[1])
    s = float(value[2])
    return d + (m / 60.0) + (s / 3600.0)

def get_location_address(latitude, longitude):
    location = geolocator.reverse((latitude, longitude), language='en', exactly_one=True)
    if location:
        return location.address
    return None

def get_image_details(our_image):
    exif_data = our_image._getexif()
    if not exif_data:
        return None
    exif = {
        ExifTags.TAGS.get(k, k): v
        for k, v in exif_data.items()
        if k in ExifTags.TAGS
    }
    important_exif = {key: exif.get(key) for key in ["Make", "Model", "DateTime", "Software", "GPSInfo"]}
    gps_info = important_exif.get("GPSInfo", {})
    if gps_info:
        latitude = convert_to_degrees(gps_info[2])
        lat_ref = gps_info[1]
        if lat_ref == 'S':
            latitude = -latitude

        longitude = convert_to_degrees(gps_info[4])
        lon_ref = gps_info[3]
        if lon_ref == 'W':
            longitude = -longitude

        important_exif['GPSCoordinates'] = (latitude, longitude)
        address = get_location_address(latitude, longitude)
        if address:
            important_exif['Address'] = address
        del important_exif['GPSInfo']
        
        width, height = our_image.size
        important_exif['Width'] = width
        important_exif['Height'] = height

    return important_exif

def plot_gps_on_map(lat, lon):
    gmap = gmplot.GoogleMapPlotter(lat, lon, 13)
    gmap.marker(lat, lon, 'red')
    temp_dir = tempfile.gettempdir()
    map_filename = os.path.join(temp_dir, "mymap.html")
    gmap.draw(map_filename)
    return map_filename

def main():
    st.title('EDIT GAMBAR HERE')
    st.text('GASKAAANNN')

    activities = ['Detection', 'About']
    choice = st.sidebar.selectbox('Select Activity', activities)

    if choice == 'Detection':
        st.subheader('Face Detection')
        image_file = st.file_uploader('Upload Image', type=['jpg', 'png', 'jpeg','png'])

        if image_file is not None:
            our_image = Image.open(image_file)
            st.text('Original Image')
            st.image(our_image)

            enhance_type = st.sidebar.radio("Enhance type", ['Original', 'Gray-scale', 'Contrast', 'Brightness', 'Blurring', 'Sharpness', 'Clustering', 'Image Details'])

            if enhance_type == 'Gray-scale':
                img = np.array(our_image.convert('RGB'))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                st.image(gray, channels="GRAY")
            elif enhance_type == "Contrast":
                rate = st.sidebar.slider("Contrast", 0.5, 6.0)
                enhancer = ImageEnhance.Contrast(our_image)
                enhanced_img = enhancer.enhance(rate)
                st.image(enhanced_img)
            elif enhance_type == "Brightness":
                rate = st.sidebar.slider("Brightness", 0.0, 8.0)
                enhancer = ImageEnhance.Brightness(our_image)
                enhanced_img = enhancer.enhance(rate)
                st.image(enhanced_img)
            elif enhance_type == "Blurring":
                rate = st.sidebar.slider("Blurring", 0.0, 7.0)
                blurred_img = cv2.GaussianBlur(np.array(our_image), (15, 15), rate)
                st.image(blurred_img)
            elif enhance_type == "Sharpness":
                rate = st.sidebar.slider("Sharpness", 0.0, 14.0)
                enhancer = ImageEnhance.Sharpness(our_image)
                enhanced_img = enhancer.enhance(rate)
                st.image(enhanced_img)
            elif enhance_type == "Clustering":
                k = st.sidebar.slider("Number of clusters (k)", 1, 20, 5)
                clustered_img = kmeans_clustering(our_image, k)
                st.image(clustered_img)
            elif enhance_type == "Image Details":
                exif_details = get_image_details(our_image)
                st.subheader('Image Details')
                if exif_details:
                    for key, value in exif_details.items():
                        if key == 'GPSCoordinates':
                            st.write(f"**{key}:** {value}")
                        elif key == 'Width' or key == 'Height':
                            st.write(f"**{key}:** {value} px")
                        else:
                            st.write(f"**{key}:** {value}")
                    if 'GPSCoordinates' in exif_details:
                        lat, lon = exif_details['GPSCoordinates']
                        st.write(f"**Google Maps Coordinates:** [View on Google Maps](https://www.google.com/maps?q={lat},{lon})")
                        map_file = plot_gps_on_map(lat, lon)
                        st.components.v1.html(open(map_file, 'r').read(), height=600)
                else:
                    st.write("No EXIF data found.")
            else:
                st.image(our_image, width=300)

        tasks = ["Faces", "Eyes", "Cartoonize"]
        feature_choice = st.sidebar.selectbox("Find features", tasks)
        if st.button("Process"):
            if feature_choice == "Faces":
                result_img, result_face = detect_faces(our_image)
                st.image(result_img)
                st.success("{} Wajah Terdeteksi".format(len(result_face)))

            if feature_choice == "Eyes":
                result_img, result_eye = detect_eye(our_image)
                st.image(result_img)
                st.success("{} Mata Terdeteksi".format(len(result_eye)))

            elif feature_choice == "Cartoonize":
                result_img = cartoonize_image(our_image)
                st.image(result_img)

    elif choice == 'About':
        st.subheader('About Developer')
        st.markdown('Built with Streamlit by Ahmad Hapizhudin')
        st.text('My name is Ahmad Hapizhudin')

if __name__ == "__main__":
    main()
