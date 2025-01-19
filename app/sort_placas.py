import numpy as np
import time

def sort_license_plate(bounding_boxes, characters,confidences, plate_coords):

    if plate_coords[2]-plate_coords[0] > 1.3*(plate_coords[3]-plate_coords[1]):
        return ''.join(characters)

    # Filter out boxes with x1 difference <= 2 and keep the highest confidence
    filtered_boxes = []
    for i, (box1, char1, conf1) in enumerate(zip(bounding_boxes, characters, confidences)):
        keep = True
        for j, (box2, char2, conf2) in enumerate(zip(bounding_boxes, characters, confidences)):
            if i != j and abs(box1[0] - box2[0]) <= 2 and abs(box1[1] - box2[1]) <= 2:
                if conf1 < conf2:
                    keep = False
                    break
        if keep:
            filtered_boxes.append((box1, char1, conf1))

    bounding_boxes, characters, confidences = zip(*filtered_boxes)

    # Sort by y-coordinate
    sorted_boxes = sorted(zip(bounding_boxes, characters), key=lambda b: b[0][1])
    
    # Group by y-coordinates
    y_positions = [box[0][1] for box in sorted_boxes]
    avg_y = np.mean(y_positions)

    straight_line = True
    for box in sorted_boxes:
        if max(-box[0][1],box[0][1]) > 0.2*avg_y:
            straight_line = False
    
    if straight_line:
        license_plate = ''.join(characters)
        print("Case 1")
        return license_plate if len(license_plate)==7 else None
    
    
    # Split into potential rows
    top_row = [(box, char) for box, char in sorted_boxes if box[1] < avg_y]
    bottom_row = [(box, char) for box, char in sorted_boxes if box[1] >= avg_y]

    # Ensure top_row and bottom_row are sorted by x-coordinate
    top_row = sorted(top_row, key=lambda b: b[0][0])
    bottom_row = sorted(bottom_row, key=lambda b: b[0][0])

    # Check if all boxes in top_row have x1 greater than all boxes in bottom_row
    if all(box[0][0] > bottom_row[-1][0][0] for box in top_row):
        license_plate =  ''.join([char for _, char in bottom_row])+''.join([char for _, char in top_row])
        print("Case 2")
        return license_plate if len(license_plate)==7 else None
    elif all(box[0][0] < top_row[0][0][0] for box in bottom_row):
        license_plate = ''.join([char for _, char in top_row])+''.join([char for _, char in bottom_row]) 
        print("Case 3")
        return license_plate if len(license_plate)==7 else None
    #print(top_row)
    #print(bottom_row)

    # Classify based on the number and type of characters
    if len(top_row) == 3 and np.mean([box[1] for box, _ in top_row]) < np.mean([box[1] for box, _ in bottom_row])- 100:
        if len(bottom_row) == 4:
            top_row = sorted(top_row, key=lambda b: b[0][0])
            bottom_row = sorted(bottom_row, key=lambda b: b[0][0])
            license_plate = ''.join([char for _, char in top_row]) + ''.join([char for _, char in bottom_row])
            print("Case 4")
            return license_plate if len(license_plate)==7 else None
    elif len(sorted_boxes) == 7:
        top_row = sorted(top_row, key=lambda b: b[0][0])
        bottom_row = sorted(bottom_row, key=lambda b: b[0][0])
        if np.mean([box[0] for box, _ in top_row]) < np.mean([box[0] for box, _ in bottom_row]):
            license_plate = ''.join([char for _, char in top_row]) + ''.join([char for _, char in bottom_row])
        else:
            license_plate = ''.join([char for _, char in bottom_row]) + ''.join([char for _, char in top_row])
        print("Case 5")
        return license_plate if len(license_plate)==7 else None
    
    print("Case 6")
    return ''.join(characters)
    
    #if len(characters)==7 else None


def time_counter_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds")
        return result
    return wrapper

