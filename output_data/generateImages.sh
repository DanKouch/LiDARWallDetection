# Display points
python3 ../scripts/identify-walls/displayLines.py  <(grep "frame_0" ehall_1800_back/cpu-points-ehall_1800_back.csv) --title="CPU Implementation - Back of 1800 Engineering Hall (Frame 0)" --outFile=images/cpu_points_ehall_1800_back.png
python3 ../scripts/identify-walls/displayLines.py  <(grep "frame_0" ehall_1800_back/gpu-points-ehall_1800_back.csv) --title="GPU Implementation - Back of 1800 Engineering Hall (Frame 0)" --outFile=images/gpu_points_ehall_1800_back.png

python3 ../scripts/identify-walls/displayLines.py  <(grep "frame_0" ehall_1800_front/cpu-points-ehall_1800_front.csv) --title="CPU Implementation - Front of 1800 Engineering Hall (Frame 0)" --outFile=images/cpu_points_ehall_1800_front.png
python3 ../scripts/identify-walls/displayLines.py  <(grep "frame_0" ehall_1800_front/gpu-points-ehall_1800_front.csv) --title="GPU Implementation - Front of 1800 Engineering Hall (Frame 0)" --outFile=images/gpu_points_ehall_1800_front.png

python3 ../scripts/identify-walls/displayLines.py  <(grep "frame_0" ehall_hallway/cpu-points-ehall_hallway.csv) --title="CPU Implementation - Engineering Hall Second Floor Hallway (Frame 0)" --outFile=images/cpu_points_ehall_hallway.png
python3 ../scripts/identify-walls/displayLines.py  <(grep "frame_0" ehall_hallway/gpu-points-ehall_hallway.csv) --title="GPU Implementation - Engineering Hall Second Floor Hallway (Frame 0)" --outFile=images/gpu_points_ehall_hallway.png

# Display overlayed
python3 ../scripts/identify-walls/displayOverlayed.py  <(grep "frame_0" ehall_1800_back/cpu-indicies-ehall_1800_back.csv) ../sample_input/ehall_1800_back/bin/frame_0.zeroDeg.bin --title="CPU Implementation - Back of 1800 Engineering Hall (Frame 0)" --outFile=images/cpu_overlayed_ehall_1800_back.png
python3 ../scripts/identify-walls/displayOverlayed.py  <(grep "frame_0" ehall_1800_back/gpu-indicies-ehall_1800_back.csv) ../sample_input/ehall_1800_back/bin/frame_0.zeroDeg.bin --title="GPU Implementation - Back of 1800 Engineering Hall (Frame 0)" --outFile=images/gpu_overlayed_ehall_1800_back.png

python3 ../scripts/identify-walls/displayOverlayed.py  <(grep "frame_0" ehall_1800_front/cpu-indicies-ehall_1800_front.csv) ../sample_input/ehall_1800_front/bin/frame_0.zeroDeg.bin --title="CPU Implementation - Front of 1800 Engineering Hall (Frame 0)" --outFile=images/cpu_overlayed_ehall_1800_front.png
python3 ../scripts/identify-walls/displayOverlayed.py  <(grep "frame_0" ehall_1800_front/gpu-indicies-ehall_1800_front.csv) ../sample_input/ehall_1800_front/bin/frame_0.zeroDeg.bin --title="GPU Implementation - Front of 1800 Engineering Hall (Frame 0)" --outFile=images/gpu_overlayed_ehall_1800_front.png

python3 ../scripts/identify-walls/displayOverlayed.py  <(grep "frame_0" ehall_hallway/cpu-indicies-ehall_hallway.csv) ../sample_input/ehall_hallway/bin/frame_0.zeroDeg.bin --title="CPU Implementation - Engineering Hall Second Floor Hallway (Frame 0)" --outFile=images/cpu_overlayed_ehall_hallway.png
python3 ../scripts/identify-walls/displayOverlayed.py  <(grep "frame_0" ehall_hallway/gpu-indicies-ehall_hallway.csv) ../sample_input/ehall_hallway/bin/frame_0.zeroDeg.bin --title="GPU Implementation - Engineering Hall Second Floor Hallway (Frame 0)" --outFile=images/gpu_overlayed_ehall_hallway.png


# Display points
python3 ../scripts/identify-walls/displayPoints.py ../sample_input/ehall_1800_back/bin/frame_0.zeroDeg.bin --title="Collected Points - Back of 1800 Engineering Hall (Frame 0)" --outFile=images/collected_points_ehall_1800_back.png

python3 ../scripts/identify-walls/displayPoints.py ../sample_input/ehall_1800_front/bin/frame_0.zeroDeg.bin --title="Collected Points - Front of 1800 Engineering Hall (Frame 0)" --outFile=images/collected_points_ehall_1800_front.png

python3 ../scripts/identify-walls/displayPoints.py  ../sample_input/ehall_hallway/bin/frame_0.zeroDeg.bin --title="Collected Points - Engineering Hall Second Floor Hallway (Frame 0)" --outFile=images/collected_points_ehall_hallway.png
