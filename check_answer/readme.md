# В папке check_answer находятся следующие файлы:<br>

- export_json_csv_from_api.py - скрипт, который выгружает json полученный в api и преобразует его в csv
- api_response.json - json полученный в api
- predictions_from_api_from_json.csv - преобразованный csv из json полученный в api
- test_predictions.csv - csv с предсказаниями выгруженный из юпитер ноутбука
- test.py - скрипт для сравнения ответов 

Была сделана попытка сравнить test_predictions.csv - csv с предсказаниями выгруженный из юпитер ноутбука, как эталонный и predictions_from_api_from_json.csv - преобразованный csv из json полученный в api, как некий студенческий вариант, но безуспешно <br>

В test.py добавлены строки для сранения файлов и закомментированы исходные строки кода<br>

    parser.add_argument("--student", type=str, required=True, default="predictions_from_api_from_json.csv", help="path to students answer")
    parser.add_argument("--correct", type=str, default="test_predictions.csv", help="path to correct answers")
    #parser.add_argument("--student", type=str, required=True, help="path to students answer")
    #parser.add_argument("--correct", type=str, default="correct_answers.csv", help="path to correct answers")