import json
import os

import settings


### Data preprocess
def preprocess():

    # image filename change (replace blank with under score)
    for i, filename in enumerate(
        os.listdir(
            os.path.abspath(os.path.join(settings.PATH_ROOT, settings.PATH_IMAGE))
        )
    ):
        new_filename = filename.replace(' ', '_')
        os.rename(
            os.path.abspath(
                os.path.join(settings.PATH_ROOT, settings.PATH_IMAGE, filename)
            ),
            os.path.abspath(
                os.path.join(settings.PATH_ROOT, settings.PATH_IMAGE, new_filename)
            ),
        )

    # Annotations format conversion
    with open(
        os.path.abspath(
            os.path.join(
                settings.PATH_ROOT,
                settings.PATH_ANNOTATIONS,
                settings.FILENAME_ANNOTATIONS,
            )
        ),
        'r',
    ) as f:
        data_json = json.loads(f.read())

    targets = {}

    for key in data_json.keys():
        find_idx = key.find('.jpg')
        new_key = key[: find_idx + 4].replace(' ', '_')
        targets[new_key] = {}
        targets[new_key]['boxes'] = []
        targets[new_key]['labels'] = []

        for i in data_json[key]['regions']:
            if data_json[key]['regions'][i]['shape_attributes']['name'] != 'polygon':
                print(
                    '{}, {} is not polygon but {}'.format(
                        key, i, data_json[key]['regions'][i]['shape_attributes']['name']
                    )
                )

            if (
                data_json[key]['regions'][i]['region_attributes']['vehicle_type']
                == 'car'
            ):
                label = 1
            else:
                print(
                    '{}, {} is not car but {}'.format(
                        key,
                        i,
                        data_json[key]['regions'][i]['region_attributes'][
                            'vehicle_type'
                        ],
                    )
                )
                label = 2

            list_x = data_json[key]['regions'][i]['shape_attributes']['all_points_x']
            list_y = data_json[key]['regions'][i]['shape_attributes']['all_points_y']
            x_min = min(list_x)
            y_min = min(list_y)
            x_max = max(list_x)
            y_max = max(list_y)

            targets[new_key]['labels'].append(label)
            targets[new_key]['boxes'].append([x_min, y_min, x_max, y_max])

    count_instances = 0
    for target in targets:
        count_instances += len(targets[target]['labels'])
    print('Number of instances: {}'.format(count_instances))

    ### Save new annotation format
    path_target_file = os.path.abspath(
        os.path.join(
            settings.PATH_ROOT, settings.PATH_TARGETS, settings.FILENAME_TARGETS
        )
    )
    with open(path_target_file, 'w') as file:
        json.dump(targets, file)

    path_preprocessed_data = {
        'images': os.path.abspath(
            os.path.join(settings.PATH_ROOT, settings.PATH_IMAGE)
        ),
        'targets': path_target_file,
    }

    return path_preprocessed_data
