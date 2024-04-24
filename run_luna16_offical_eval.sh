#! /bin/bash

# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

python ./luna16_post_combine_cross_fold_results.py \
	-i ./trained_results/result_luna16_fold9.json \
	-o ./trained_results/result_luna16_fold9.csv

python ./evaluationScript/noduleCADEvaluationLUNA16.py \
	./evaluationScript/annotations/new_annotations.csv  \
	./evaluationScript/annotations/new_annotations_excluded.csv \
	./LUNA16_datasplit/fold9.csv \
	./trained_results/result_luna16_fold9.csv \
	./trained_results

