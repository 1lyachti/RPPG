from pathlib import Path
from scipy.signal import butter, sosfiltfilt


import matplotlib.pyplot as plt
import numpy as np


data_path = Path(__file__).with_name("rgb_adaptive.csv")
data = np.loadtxt(data_path, delimiter=",", skiprows=1, ndmin=2)


if len(data) < 2:
    raise SystemExit("Недостатньо даних для побудови графіка")

time = data[:, 0]
r = data[:, 1]
g = data[:, 2] 
b = data[:, 3]

intervals = np.diff(time)
largest_gap_indices = np.argsort(intervals)[-5:][::-1]

print("\nНайбільші проміжки між вимірюваннями:")

for index in largest_gap_indices:
    print(
        f"{time[index]:.3f}–{time[index + 1]:.3f} с: "
        f"{intervals[index]:.3f} с"
    )
gap_index = np.argmax(intervals)
print(
    f"Найбільший проміжок між вимірюваннями: "
    f"{time[gap_index]:.3f}–{time[gap_index + 1]:.3f} с"
)
if np.any(intervals <= 0):
    raise SystemExit("Часові мітки повинні бути строго зростаючими")

duration = time[-1] - time[0]
sampling_rate = (len(time) - 1) / duration

print(f"Зразки: {len(time)}")
print(f"Тривалість: {duration:.2f} с")
print(f"Середня частота дискретизації: {sampling_rate:.2f} Гц")
print(f" Мінімальний інтервал: {np.min(intervals):.3f} с")
print(f"Медіанний інтервал: {np.median(intervals):.3f} с")
print(f"Максимальний інтервал: {np.max(intervals):.3f} с")



uniform_time = np.linspace(time [0], time[-1], len(time))
uniform_r = np.interp(uniform_time, time, r)
uniform_g = np.interp(uniform_time, time, g)
uniform_b = np.interp(uniform_time, time, b)

uniform_step = uniform_time[1] - uniform_time[0]

print(f"Рівномірний крок: {uniform_step:.4f} с  ")
print (f"Рівномірна частота інтерполяції: {1 / uniform_step:.2f} Гц")
plt.figure(figsize=(12, 5))
plt.plot(time, g, ".", color="gray", label="Початковий G")
plt.plot(uniform_time, uniform_g, color="green", label="Рівномірний G")
plt.xlabel("Час (с)")
plt.ylabel("Інтенсивність (0-255)")
plt.title("Адаптивна обробка RGB сигналу")
plt.legend()
plt.grid(True)
plt.show()


sampling_rate = 1 / uniform_step
window_size = int(np.ceil(1.6 * sampling_rate))

start_index = np.searchsorted(uniform_time, 35.0)
end_index = start_index + window_size

if end_index > len(uniform_time):
    raise SystemExit("Недостатньо даних для обробки вікна після 35 секунд")

window_time = uniform_time[start_index:end_index]

rgb_window = np.column_stack([
    uniform_r[start_index:end_index],
    uniform_g[start_index:end_index],
    uniform_b[start_index:end_index],
])

channel_means = rgb_window.mean(axis=0)
normalized_rgb = rgb_window / channel_means - 1

print(f"Розмір вікна: {window_size} зразків")
print(f"Середні значення RGB вікна: {channel_means}")

plt.figure(figsize=(10, 4))
plt.plot(window_time, normalized_rgb[:, 0], color="red", label="R")
plt.plot(window_time, normalized_rgb[:, 1], color="green", label="G")
plt.plot(window_time, normalized_rgb[:, 2], color="blue", label="B")
plt.xlabel("Час (с)")
plt.ylabel("Відносна зміна")
plt.title("RGB після нормалізації в одному вікні")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

rn = normalized_rgb[:, 0]
gn = normalized_rgb[:, 1]
bn = normalized_rgb[:, 2]

s1 = gn - bn
s2 = gn + bn - 2 * rn

s2_std = np.std(s2)

if s2_std < 1e-12:
    raise SystemExit("В цьому вікні недостатньо змін для розрахунку POS")

alpha = np.std(s1) / s2_std
pos_window = s1 + alpha * s2
pos_window = pos_window - pos_window.mean()

print(f"Коефіцієнт alpha: {alpha:.3f}")

plt.figure(figsize=(10, 4))
plt.plot(window_time, pos_window, color="black", label="POS")
plt.xlabel("Час (с)")
plt.ylabel("Відносна амплітуда")
plt.title("POS в одному часовому вікні")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

rgb_uniform = np.column_stack([
    uniform_r,
    uniform_g,
    uniform_b,
])

pos_sum = np.zeros(len(uniform_time))
window_counts = np.zeros(len(uniform_time))

for start_index in range(len(uniform_time) - window_size + 1):
    end_index = start_index + window_size

    rgb_window = rgb_uniform[start_index:end_index]
    channel_means = rgb_window.mean(axis=0)

    if np.any(channel_means <= 0):
        continue

    normalized = rgb_window / channel_means - 1

    rn = normalized[:, 0]
    gn = normalized[:, 1]
    bn = normalized[:, 2]

    s1 = gn - bn
    s2 = gn + bn - 2 * rn

    s2_std = np.std(s2)

    if s2_std < 1e-12:
        continue

    alpha = np.std(s1) / s2_std
    window_signal = s1 + alpha * s2
    window_signal = window_signal - window_signal.mean()

    pos_sum[start_index:end_index] += window_signal
    window_counts[start_index:end_index] += 1

valid = window_counts > 0
pos_signal = np.full(len(uniform_time), np.nan)
pos_signal[valid] = pos_sum[valid] / window_counts[valid]

plt.figure(figsize=(12, 4))
plt.plot(uniform_time, pos_signal, color="black")
plt.xlabel("Час (с)")
plt.ylabel("Відносна амплітуда")
plt.title("POS для всього запису")
plt.grid(True)
plt.tight_layout()
plt.show()

if not np.all(np.isfinite(pos_signal)):
    raise SystemExit("В POS є пропуски: перед фільтрацією потрібно перевірити вікна")

sos = butter(
    3,
    [0.7, 3.0],
    btype="bandpass",
    fs=sampling_rate,
    output="sos",
)

filtered_pos = sosfiltfilt(sos, pos_signal)

section = (uniform_time >= 35) & (uniform_time <= 50)

plt.figure(figsize=(12, 4))
plt.plot(
    uniform_time[section],
    pos_signal[section],
    color="gray",
    alpha=0.6,
    label="POS до фільтрації",
)
plt.plot(
    uniform_time[section],
    filtered_pos[section],
    color="black",
    label="POS після фільтрації",
)
plt.xlabel("Час (с)")
plt.ylabel("Відносна амплітуда")
plt.title("Порівняння POS до та після фільтрації")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

segment = filtered_pos[section]
segment = segment - segment.mean()

window = np.hanning(len(segment))
windowed_segment = segment * window

spectrum = np.fft.rfft(windowed_segment)
frequencies = np.fft.rfftfreq(
    len(windowed_segment),
    d=uniform_step,
)

power = np.abs(spectrum) ** 2

search_band = (frequencies >= 0.7) & (frequencies <= 3.0)
band_frequencies = frequencies[search_band]
band_power = power[search_band]

peak_index = np.argmax(band_power)
peak_frequency = band_frequencies[peak_index]
estimated_bpm = peak_frequency * 60

frequency_step = sampling_rate / len(segment)

print(f"Пікова частота: {peak_frequency:.3f} Гц")
print(f"Попередня оцінка: {estimated_bpm:.1f} уд/хв")
print(f"Крок спектру: {frequency_step * 60:.1f} уд/хв")

plt.figure(figsize=(10, 4))
plt.plot(band_frequencies * 60, band_power, color="black")
plt.axvline(
    estimated_bpm,
    color="red",
    linestyle="--",
    label=f"Пік: {estimated_bpm:.1f} уд/хв",
)
plt.xlabel("Частота (уд/хв)")
plt.ylabel("Спектральна потужність, умовні одиниці")
plt.title("Спектр POS на ділянці 35–50 с")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

intervals_to_check = [
    (5, 20),
    (20, 35),
    (35, 50),
]

print("\nОцінки для окремих ділянок")
print("Ділянка, с | Частота, Гц | Оцінка, уд/хв")

for start_time, end_time in intervals_to_check:
    if start_time < uniform_time[0] or end_time > uniform_time[-1]:
        print(
            f"{start_time}–{end_time} с: "
            "пропущено — запис не охоплює всю ділянку"
        )
        continue

    selected = (
        (uniform_time >= start_time)
        & (uniform_time <= end_time)
    )

    segment = filtered_pos[selected]

    if len(segment) < 3:
        print(f"{start_time}–{end_time}: недостатньо даних")
        continue

    segment = segment - segment.mean()
    windowed_segment = segment * np.hanning(len(segment))

    spectrum = np.fft.rfft(windowed_segment)
    frequencies = np.fft.rfftfreq(
        len(segment),
        d=uniform_step,
    )
    power = np.abs(spectrum) ** 2

    search_band = (frequencies >= 0.7) & (frequencies <= 3.0)
    band_frequencies = frequencies[search_band]
    band_power = power[search_band]

    peak_index = np.argmax(band_power)
    peak_frequency = band_frequencies[peak_index]
    estimated_bpm = peak_frequency * 60

    print(
        f"{start_time:2}–{end_time:<2}      | "
        f"{peak_frequency:11.3f} | "
        f"{estimated_bpm:13.1f}"
    )