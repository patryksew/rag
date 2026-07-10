from src.models import StudentSearchResults, RagDataset, UnansweredQuestion, \
    MinimalSource, AnsweredQuestion


class Evaluator:
    """
    Evaluator for assessing the quality of retrieval results.
    """

    @staticmethod
    def _get_question_by_id(q_id: str,
                            data: list[UnansweredQuestion | AnsweredQuestion]) -> AnsweredQuestion:
        """
        Retrieve a question from the dataset by its ID.

        Args:
            q_id (str): The question ID.
            data (list[UnansweredQuestion | AnsweredQuestion]): List of questions in the dataset.

        Returns:
            AnsweredQuestion: The matched answered question.

        Raises:
            ValueError: If the question with the given ID is not found.
        """
        for q in data:
            if not isinstance(q, AnsweredQuestion):
                continue
            if q.question_id == q_id:
                return q

        raise ValueError(f"Question with id {q_id} not found in the dataset.")

    @staticmethod
    def _iou(student: MinimalSource, truth: MinimalSource) -> float:
        """
        Calculate the Intersection over Union (IoU) between a retrieved source and ground truth.

        Args:
            student (MinimalSource): The retrieved source to evaluate.
            truth (MinimalSource): The ground truth source.

        Returns:
            float: The IoU score.
        """
        if student.first_character_index >= truth.last_character_index:
            return 0

        if student.last_character_index <= truth.first_character_index:
            return 0

        intersection = min(student.last_character_index, truth.last_character_index) - max(
            student.first_character_index, truth.first_character_index)

        union = max(student.last_character_index, truth.last_character_index) - min(
            student.first_character_index, truth.first_character_index)

        return intersection / union

    @staticmethod
    def _are_sources_valid(student: list[MinimalSource], truth: list[MinimalSource]) -> bool:
        """
        Check if any of the retrieved sources adequately match the ground truth sources.

        Args:
            student (list[MinimalSource]): The retrieved sources to evaluate.
            truth (list[MinimalSource]): The ground truth sources.

        Returns:
            bool: True if at least one retrieved source is valid, False otherwise.
        """
        for student_source in student:
            for truth_source in truth:
                if student_source.file_path == truth_source.file_path:
                    if Evaluator._iou(student_source, truth_source) >= 0.05:
                        return True

        return False

    @staticmethod
    def evaluate(student_search_results_path: str, dataset_path: str) -> float:
        """
        Evaluate and return the recall@k metric for the retrieved search results against a dataset.

        Args:
            student_search_results_path (str): The path to the JSON file with the search results.
            dataset_path (str): The path to the ground truth dataset JSON file.

        Returns:
            float: The calculated recall@k metric.
        """
        with open(student_search_results_path, "r") as f:
            student_search_results = StudentSearchResults.model_validate_json(f.read())

        with open(dataset_path, "r") as f:
            dataset = RagDataset.model_validate_json(f.read())

        sources_ok = 0

        for student in student_search_results.search_results:
            q_id = student.question_id
            data = Evaluator._get_question_by_id(q_id, dataset.rag_questions)
            if Evaluator._are_sources_valid(student.retrieved_sources, data.sources):
                sources_ok += 1

        return sources_ok / len(dataset.rag_questions)
