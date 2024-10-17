# Tic Tac Toe dengan Konsep AlphaGo
## UTS Kelompok 2
- Nazir Mahmudi Lubis (G1A021013)
- Rosalia Dina Marina Sipahutar (G1A021017)
- Muhammad Fachrurozi (G1A021018)

[](https://github.com/user-attachments/assets/0a9a97a3-fa04-4e7b-bbac-a66440d86c68)

Repositori ini berisi kode Python  dalam simulasi permainan tic tac toe sederhana dengan implementasi AlphaGo, Reinforcement Learning (RL), dan Monte Carlo Tree Search (MCTS)

## Reinforcement Learning
Reinforcement Learning (RL) adalah pendekatan Machine Learning di mana agen belajar untuk membuat keputusan dengan cara mencoba berbagai tindakan dan menerima umpan balik dalam bentuk penghargaan atau hukuman. AlphaGo mengintegrasikan beberapa konsep dari RL:
  - **Agent:** Entitas yang membuat keputusan (dalam hal ini, AI).
  - **Environment:** Lingkungan tempat agent beroperasi (papan permainan Tic-Tac-Toe).
  - **State:** Situasi saat ini dari environment (posisi batu di papan).
  - **Action:** Langkah yang diambil oleh agent (memilih koordinat untuk menempatkan batu).
  - **Reward:** Umpan balik dari environment setelah tindakan diambil (1 untuk kemenangan, -1 untuk kekalahan, dan 0 untuk permainan yang berlanjut).

## AlphaGo dan Reinforcement Learning
AlphaGo adalah sistem AI yang dikembangkan oleh DeepMind untuk bermain Go, permainan papan yang jauh lebih kompleks daripada Tic-Tac-Toe. AlphaGo menggunakan pendekatan berbasis Reinforcement Learning, di mana AI belajar dari pengalaman untuk meningkatkan strategi bermainnya. Dengan menggunakan kombinasi dari Neural Networks untuk memprediksi langkah terbaik dan nilai dari posisi papan, serta MCTS untuk menjelajahi kemungkinan langkah, AlphaGo dapat mengalahkan pemain Go terbaik di dunia.

## Monte Carlo Tree Search (MCTS)
MCTS adalah algoritma yang digunakan untuk membuat keputusan dalam permainan dengan banyak kemungkinan langkah. MCTS melakukan simulasi acak dari permainan yang tersisa untuk mengevaluasi langkah-langkah potensial dan membangun pohon keputusan. MCTS terdiri dari empat langkah utama:
1. **Selection:** Memilih node dari pohon keputusan yang sudah ada.
2. **Expansion:** Menambahkan node baru untuk langkah yang tersedia.
3. **Simulation:** Mensimulasikan permainan secara acak dari node baru hingga akhir permainan.
4. **Backpropagation:** Mengupdate informasi tentang node di pohon berdasarkan hasil simulasi.
MCTS memungkinkan AI untuk mengeksplorasi langkah-langkah yang lebih baik dengan efisien tanpa harus mengeksplorasi semua kemungkinan langkah di awal.

## Game Tic Tac Toe dan Reinforcement Learning
Game ini menerapkan konsep Reinforcement Learning dengan memanfaatkan MCTS untuk mengevaluasi langkah-langkah yang mungkin diambil oleh AI. AI berfungsi sebagai agent yang berinteraksi dengan environment (papan permainan). Melalui simulasi dan pembelajaran dari hasil permainan, AI dapat meningkatkan kemampuannya untuk memilih langkah terbaik di masa depan. Pada setiap langkah, AI mengevaluasi posisi saat ini dan menentukan langkah optimal berdasarkan pengalaman yang telah dikumpulkan, sehingga membuatnya semakin pintar dalam bermain.

## Cara Bermain
1. Setelah permainan dimulai, user dapat mengklik pada papan untuk menempatkan simbol (X).
2. AI akan secara otomatis membuat langkah berikutnya.
3. Permainan berakhir ketika salah satu pemain berhasil menyusun tiga simbol dalam satu garis (horizontal, vertikal, atau diagonal).


## Instalasi
Untuk menjalankan proyek ini, Perlu menginstal beberapa dependensi Python. Pastikan Python versi 3.x telah terinstal di sistem. Berikut adalah dependensi yang diperlukan: 
- **NumPy** 
- **Matplotlib** 
- **TensorFlow**
