"""
Aplikasi Color Picker menggunakan K-Means Clustering

Aplikasi ini menggunakan algoritma K-Means untuk mengekstrak warna dominan 
dari gambar yang diunggah. Setiap piksel dalam gambar direpresentasikan sebagai 
titik dalam ruang warna RGB 3D, dan K-Means digunakan untuk mengelompokkan 
piksel-piksel tersebut ke dalam cluster warna yang mirip.

Fitur utama:
- K-Means clustering dengan K-means++ initialization
- Visualisasi 3D interaktif menggunakan Plotly
- Ekstraksi palet warna dalam format RGB dan Hex
- Interface pengguna yang intuitif dengan Streamlit

Nama: Bagas Diatama Wardoyo
"""

import streamlit as st
from PIL import Image
import numpy as np
import plotly.graph_objects as go

def kmeans(pixels, k=5, max_iters=100, tolerance=1e-4):
    """
    Implementasi algoritma K-Means clustering dengan K-means++ initialization.
    
    Algoritma ini mengelompokkan piksel gambar ke dalam k cluster berdasarkan
    kemiripan warna dalam ruang RGB. Menggunakan K-means++ untuk inisialisasi
    centroid yang lebih baik dibandingkan inisialisasi acak.
    
    Args:
        pixels (np.ndarray): Array 2D berisi nilai RGB piksel dengan shape (n_pixels, 3)
        k (int, optional): Jumlah cluster yang diinginkan. Default: 5
        max_iters (int, optional): Maksimum iterasi algoritma. Default: 100
        tolerance (float, optional): Threshold konvergensi untuk menghentikan iterasi. Default: 1e-4
    
    Yields:
        tuple: (centroids, labels, iteration) untuk setiap iterasi
            - centroids (np.ndarray): Posisi centroid dengan shape (k, 3)
            - labels (np.ndarray): Label cluster untuk setiap piksel dengan shape (n_pixels,)
            - iteration (int): Nomor iterasi saat ini
    
    Note:
        Fungsi ini adalah generator yang menghasilkan hasil di setiap iterasi,
        memungkinkan visualisasi proses clustering secara real-time.
    """
    n_pixels = len(pixels)
    centroids = np.zeros((k, 3))
    
    # K-means++ Initialization: Pilih centroid pertama secara acak
    centroids[0] = pixels[np.random.choice(n_pixels)]
    
    # Pilih centroid selanjutnya dengan probabilitas proporsional terhadap jarak kuadrat
    for c_id in range(1, k):
        # Hitung jarak minimum setiap piksel ke centroid yang sudah ada
        distances = np.array([min([np.linalg.norm(pixel - centroid)**2 
                                 for centroid in centroids[:c_id]]) 
                             for pixel in pixels])
        
        # Konversi jarak ke probabilitas (semakin jauh, semakin besar peluang dipilih)
        probabilities = distances / distances.sum()
        cumulative_probabilities = probabilities.cumsum()
        r = np.random.rand()
        
        # Pilih piksel berdasarkan probabilitas kumulatif
        for j, p in enumerate(cumulative_probabilities):
            if r < p:
                centroids[c_id] = pixels[j]
                break

    # Iterasi utama algoritma K-Means
    iteration = 0
    for iteration in range(max_iters):
        # Assignment Step: Assign setiap piksel ke cluster terdekat
        distances = np.linalg.norm(pixels[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # Yield hasil iterasi saat ini
        yield centroids.copy(), labels.copy(), iteration + 1

        # Update Step: Hitung centroid baru sebagai rata-rata cluster
        new_centroids = np.zeros((k, 3))
        centroid_moved = False
        
        for j in range(k):
            cluster_mask = labels == j
            cluster_pixels = pixels[cluster_mask]
            
            if len(cluster_pixels) > 0:
                # Hitung centroid baru sebagai rata-rata piksel dalam cluster
                new_centroid = np.mean(cluster_pixels, axis=0)
                
                # Cek apakah centroid bergerak signifikan
                if np.linalg.norm(new_centroid - centroids[j]) > tolerance:
                    centroid_moved = True
                    
                new_centroids[j] = new_centroid
            else:
                # Handling cluster kosong: pilih piksel terjauh dari semua centroid
                distances_to_centroids = np.min(
                    np.linalg.norm(pixels[:, np.newaxis] - centroids, axis=2), 
                    axis=1
                )
                farthest_pixel_idx = np.argmax(distances_to_centroids)
                new_centroids[j] = pixels[farthest_pixel_idx]
                centroid_moved = True
        
        # Cek konvergensi: jika tidak ada centroid yang bergerak signifikan
        if not centroid_moved:
            centroids = new_centroids
            # Hitung assignment final
            distances = np.linalg.norm(pixels[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            yield centroids, labels, iteration + 1
            break
            
        centroids = new_centroids
    
    # Yield hasil final jika mencapai maksimum iterasi
    if iteration == max_iters - 1:
        distances = np.linalg.norm(pixels[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        yield centroids, labels, iteration + 1


def create_3d_plot(pixels, centroids, labels, iteration):
    """
    Membuat visualisasi 3D interaktif dari hasil clustering K-Means.
    
    Fungsi ini membuat scatter plot 3D di mana sumbu X, Y, Z merepresentasikan
    nilai Red, Green, Blue dari setiap piksel. Setiap titik diwarnai berdasarkan
    cluster yang dimilikinya, dan centroid ditampilkan sebagai marker 'X' merah.
    
    Args:
        pixels (np.ndarray): Array piksel RGB dengan shape (n_pixels, 3)
        centroids (np.ndarray): Posisi centroid dengan shape (k, 3)
        labels (np.ndarray): Label cluster untuk setiap piksel
        iteration (str/int): Informasi iterasi untuk judul plot
    
    Returns:
        plotly.graph_objects.Figure: Object figure Plotly yang siap ditampilkan
    
    Note:
        Untuk performa yang lebih baik, fungsi ini akan mengambil sampel acak
        maksimal 2000 piksel jika dataset terlalu besar.
    """
    # Sampling untuk performa: batasi visualisasi maksimal 2000 piksel
    sample_size = 2000
    if len(pixels) > sample_size:
        sample_indices = np.random.choice(len(pixels), sample_size, replace=False)
        pixels_sample = pixels[sample_indices]
        labels_sample = labels[sample_indices]
    else:
        pixels_sample = pixels
        labels_sample = labels

    # Trace untuk titik data (piksel)
    pixel_trace = go.Scatter3d(
        x=pixels_sample[:, 0],  # Red channel
        y=pixels_sample[:, 1],  # Green channel  
        z=pixels_sample[:, 2],  # Blue channel
        mode='markers',
        marker=dict(
            size=3,
            color=labels_sample,    # Warna berdasarkan label cluster
            colorscale='Viridis',   # Skema warna yang mudah dibedakan
            opacity=0.7
        ),
        name='Piksel'
    )

    # Trace untuk centroid (pusat cluster)
    centroid_trace = go.Scatter3d(
        x=centroids[:, 0],      # Red channel centroid
        y=centroids[:, 1],      # Green channel centroid
        z=centroids[:, 2],      # Blue channel centroid
        mode='markers',
        marker=dict(
            size=10,
            symbol='x',             # Bentuk marker X untuk centroid
            color='red',            # Warna merah agar mudah terlihat
            line=dict(color='black', width=2)
        ),
        name='Centroids'
    )

    # Layout untuk plot 3D
    layout = go.Layout(
        title=f'Visualisasi K-Means Iterasi #{iteration}',
        scene=dict(
            xaxis_title='Red (R)',      # Sumbu X = nilai Red
            yaxis_title='Green (G)',    # Sumbu Y = nilai Green
            zaxis_title='Blue (B)'      # Sumbu Z = nilai Blue
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig = go.Figure(data=[pixel_trace, centroid_trace], layout=layout)
    return fig


def rgb_to_hex(rgb_color):
    """
    Konversi nilai RGB ke format hexadecimal.
    
    Args:
        rgb_color (list/tuple): Nilai RGB dalam format [R, G, B] atau (R, G, B)
    
    Returns:
        str: Kode warna hexadecimal dalam format #RRGGBB
    
    Example:
        >>> rgb_to_hex([255, 0, 0])
        '#FF0000'
    """
    r, g, b = map(int, rgb_color)
    return f"#{r:02x}{g:02x}{b:02x}"

# =============================================================================
# STREAMLIT USER INTERFACE
# =============================================================================

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Color Picker K-Means",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Header aplikasi
st.title("Color Picker menggunakan K-Means Clustering")
st.markdown("""
**Deskripsi:** Aplikasi ini menggunakan algoritma K-Means untuk mengekstrak warna dominan dari gambar. 
Setiap piksel direpresentasikan sebagai titik dalam ruang warna RGB 3D, kemudian dikelompokkan 
berdasarkan kemiripan warna.

**Cara penggunaan:**
1. Unggah gambar dalam format JPG, JPEG, atau PNG
2. Pilih jumlah warna yang ingin diekstrak menggunakan slider
3. Klik tombol "Ekstrak Palet Warna" untuk memulai proses
""")

# Widget upload file
uploaded_file = st.file_uploader(
    label="📁 Pilih gambar yang akan dianalisis",
    type=["jpg", "jpeg", "png"], 
    help="Format yang didukung: JPG, JPEG, PNG. Ukuran maksimal: 200MB"
)

# Proses jika ada file yang diunggah
if uploaded_file is not None:
    try:
        # Buka dan proses gambar
        image = Image.open(uploaded_file)
        
        # Resize gambar untuk efisiensi komputasi (100x100 piksel = 10,000 titik data)
        img_resized = image.resize((100, 100)).convert("RGB")
        pixels = np.array(img_resized).reshape(-1, 3)

        # Tampilkan gambar yang diunggah
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, caption="Gambar Original", use_column_width=True)
        with col2:
            st.image(img_resized, caption="Gambar yang Diproses (100x100)", use_column_width=True)
        
        st.markdown("---")
        
        # Kontrol jumlah warna
        num_colors = st.slider(
            label="Jumlah warna dalam palet:",
            min_value=2,
            max_value=10,
            value=5,
            step=1,
            help="Pilih berapa banyak warna dominan yang ingin diekstrak dari gambar. "
                 "Semakin banyak cluster, semakin detail hasil ekstraksi warna."
        )
        
        # Informasi untuk user
        st.info(f"**Info:** Gambar akan dianalisis menggunakan {pixels.shape[0]:,} piksel "
                f"untuk mengekstrak **{num_colors}** warna dominan.")

        # Tombol untuk memulai proses
        if st.button("Ekstrak Palet Warna", use_container_width=True, type="primary"):
            
            # Inisialisasi progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Jalankan algoritma K-Means
            kmeans_gen = kmeans(pixels, k=num_colors, max_iters=100, tolerance=1e-4)
            
            status_text.text("Menjalankan algoritma K-Means...")
            
            # Iterasi melalui hasil K-Means
            total_iterations = 0
            for centroids, labels, iteration in kmeans_gen:
                final_centroids = centroids
                final_labels = labels
                final_iteration = iteration
                total_iterations = iteration
                
                # Update progress bar (estimasi berdasarkan iterasi)
                progress = min(iteration / 20, 1.0)  # Estimasi maksimal 20 iterasi
                progress_bar.progress(progress)
            
            # Selesaikan progress bar
            progress_bar.progress(1.0)
            status_text.text(f"Selesai! K-Means konvergen dalam {total_iterations} iterasi.")
            
            # Tampilkan hasil
            st.success(f"**Berhasil!** Algoritma K-Means selesai dalam {final_iteration} iterasi "
                      f"dan berhasil mengekstrak {num_colors} warna dominan.")
            
            # Bagian visualisasi 3D
            st.markdown("### Visualisasi Clustering 3D")
            st.markdown("Plot di bawah menunjukkan distribusi piksel dalam ruang warna RGB 3D. "
                       "Setiap titik merepresentasikan satu piksel, diwarnai berdasarkan cluster-nya. "
                       "Tanda 'X' merah menunjukkan posisi centroid (pusat cluster).")
            
            with st.spinner("Membuat visualisasi 3D..."):
                fig = create_3d_plot(pixels, final_centroids, final_labels, 
                                   iteration=f"Final ({final_iteration} iterasi)")
                st.plotly_chart(fig, use_container_width=True)
            
            # Bagian palet warna
            if final_centroids is not None:
                st.markdown(f"### Palet {num_colors} Warna Dominan")
                st.markdown("Berikut adalah warna-warna dominan yang berhasil diekstrak:")
                
                # Konversi centroid ke format integer untuk RGB
                colors = final_centroids.astype(int).tolist()
                
                # Tampilkan palet warna dalam kolom
                cols = st.columns(num_colors)
                for i, color in enumerate(colors):
                    with cols[i]:
                        hex_color = rgb_to_hex(color)
                        
                        # Box warna
                        st.markdown(
                            f"""
                            <div style="
                                background-color: {hex_color}; 
                                padding: 40px; 
                                border-radius: 10px; 
                                border: 2px solid #ddd;
                                margin-bottom: 10px;
                            "></div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        # Informasi warna
                        st.code(hex_color.upper(), language=None)
                        st.caption(f"RGB: ({color[0]}, {color[1]}, {color[2]})")
                        
                        # Tambahan: persentase piksel dalam cluster ini
                        cluster_percentage = (np.sum(final_labels == i) / len(final_labels)) * 100
                        st.caption(f"📊 {cluster_percentage:.1f}% dari gambar")
            
    except Exception as e:
        st.error(f"**Error:** Terjadi kesalahan saat memproses gambar: {str(e)}")
        st.info("**Tips:** Pastikan file yang diunggah adalah gambar yang valid dan tidak corrupt.")

else:
    # Tampilan ketika belum ada file yang diunggah
    st.info("**Mulai dengan mengunggah gambar di atas!**")

    # Tambahan: contoh atau informasi algoritma
    with st.expander("Tentang Algoritma K-Means"):
        st.markdown("""
        **K-Means Clustering** adalah algoritma machine learning unsupervised yang digunakan untuk 
        mengelompokkan data ke dalam k cluster berdasarkan kemiripan fitur.
        
        **Dalam konteks color picker:**
        - Setiap piksel gambar direpresentasikan sebagai titik 3D dengan koordinat (R, G, B)
        - K-Means mengelompokkan piksel-piksel dengan warna serupa ke dalam cluster yang sama
        - Centroid setiap cluster menjadi warna representatif dari cluster tersebut
        
        **Keunggulan K-means++ initialization:**
        - Memilih centroid awal yang tersebar dengan baik
        - Mengurangi kemungkinan konvergensi ke local minimum
        - Hasil clustering yang lebih konsisten dan berkualitas
        """)