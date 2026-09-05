class AppError(Exception):
    def __init__(self, code, message, status=503):
        super().__init__(code)
        self.code, self.message, self.status = code, message, status
