class Report:
    """Get GPTCache report including time and counts for different operations."""

    def __init__(self):
        self.embedding_all_time = 0
        self.embedding_count = 0
        self.search_all_time = 0
        self.search_count = 0
        self.hint_cache_count = 0

    def embedding(self, delta_time):
        """Embedding counts and time.

        :param delta_time: additional runtime.
        """
        self.embedding_all_time += delta_time
        self.embedding_count += 1

    def search(self, delta_time):
        """Search counts and time.

        :param delta_time: additional runtime.
        """
        self.search_all_time += delta_time
        self.search_count += 1

    def average_embedding_time(self):
        """Average embedding time."""
        return round(
            self.embedding_all_time / self.embedding_count
            if self.embedding_count != 0
            else 0,
            4,
        )

    def average_search_time(self):
        return round(
            self.search_all_time / self.search_count
            if self.embedding_count != 0
            else 0,
            4,
        )

    def hint_cache(self):
        self.hint_cache_count += 1
