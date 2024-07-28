def algo(row_index, img, start_at_col=0):
    print("starting algo for row " + str(row_index) + "...")
    row_steps = []
    actual_row_index = row_index
    # for column_index in range(0, int((img2.shape[0] / 5) - 1)):
    for column_index in range(start_at_col, img.shape[0] - 1):
        block_brightnesses = dict([(0, 0), (-1, 0), (1, 0), (-2, 0), (2, 0), (-3, 0), (3, 0)])
        allowed_height_diffs = [0]
        if actual_row_index - row_index <= 25:
            allowed_height_diffs.append(-1)
            allowed_height_diffs.append(-2)
            allowed_height_diffs.append(-3)
        if row_index - actual_row_index <= 25:
            allowed_height_diffs.append(1)
            allowed_height_diffs.append(2)
            allowed_height_diffs.append(3)
        for height_diff in allowed_height_diffs:
            actual_height = actual_row_index + height_diff
            if actual_height < 0 or actual_height >= img.shape[1]:
                block_brightnesses[height_diff] = -1
                continue
            block_brightnesses[height_diff] = 255 if img[column_index, actual_height][0] == 255 else 0
        brightest = max(block_brightnesses, key=block_brightnesses.get)
        if brightest == 0 and block_brightnesses[0] == block_brightnesses[1] and len(row_steps) > 100 and sum([0 if x[0] == 0 else 1 for x in row_steps][-100:]) == 0:
            tendency = 1 if sum([x[0] for x in row_steps]) > 0 else -1
            row_steps.append((tendency, block_brightnesses[tendency]))
            actual_row_index += tendency
        else:
            row_steps.append((brightest, block_brightnesses[brightest]))
            actual_row_index += brightest
    avg_row_brightness = sum(map(lambda row:row[1], row_steps)) / (len(row_steps))
    if int(avg_row_brightness) >= 254:
        print("storing row for later!")
        return(row_index, [instruction[0] for instruction in row_steps])
    return None


