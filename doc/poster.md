# poster

## OpenPose

### Motivation

- Aim: human 2D pose estimation

- Current 2D human skeleton models: Body25, COCO, MPI.
- Body25: the most accurate, the fastest on GPU.
- OpenPose: real time, high accuracy, easy-to-use API.
- Output: keypoint coordinates, confidence levels

### Data Processing

- Scale-invariant features:
  - 8 degrees: knees, elbows, shoulders, hips
  - 25 Normalized keypoint coordinates: [0,m] x [0,n] -> [0,1] x [0,1]
  - 25 confidence levels: (0,1]

- Effectiveness: state estimation based on linear SVC ~60% on test set