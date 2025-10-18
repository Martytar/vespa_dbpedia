import json
import csv
import uuid


def csv_to_vespa_jsonl_dict(csv_file_path, jsonl_file_path):
    """
    Версия с DictReader - если в файле есть заголовки
    """

    with open(csv_file_path, 'r', encoding='utf-8') as csv_file, \
            open(jsonl_file_path, 'w', encoding='utf-8') as jsonl_file:

        # Сначала посмотрим на первую строку чтобы понять структуру
        first_line = csv_file.readline().strip()
        print(f"Первая строка: {repr(first_line)}")

        # Возвращаемся к началу
        csv_file.seek(0)

        # Пробуем прочитать как CSV с заголовками
        csv_reader = csv.DictReader(csv_file, delimiter=',', quotechar='"')

        print(f"Поля в CSV: {csv_reader.fieldnames}")

        processed_count = 0

        for row_num, row in enumerate(csv_reader, 1):

            unique_id = str(uuid.uuid4())

            # В зависимости от структуры столбцов выбираем данные
            # Предполагаем, что первый столбец игнорируем, второй - title, третий - description
            fields = list(row.values())

            if len(fields) >= 2:
                title = fields[1]
                description = fields[2] if len(fields) > 2 else ""

                # Если больше столбцов, объединяем в description
                if len(fields) > 3:
                    description = ','.join(fields[2:])
            else:
                title = fields[0] if fields else ""
                description = ""

            vespa_doc = {
                "put": f"id:news:news::{unique_id}",
                "fields": {
                    "news_id": unique_id,
                    "title": title,
                    "description": description
                }
            }

            jsonl_file.write(json.dumps(vespa_doc, ensure_ascii=False) + '\n')
            processed_count += 1

        print(f"Обработано документов: {processed_count}")
        return processed_count


# Использование
if __name__ == "__main__":
    csv_to_vespa_jsonl_dict('DBpedia\\train.csv',
                       'DBpedia\\news.jsonl')
    print("Конвертация завершена! Файл vespa_data.jsonl создан.")