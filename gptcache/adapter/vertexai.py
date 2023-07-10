from google.cloud import aiplatform_v1 as aiplatform

class VertexAIAdapter:
    def initialize(self, project, location, *args, **kwargs):
        self.project = project
        self.location = location
        self.client = aiplatform.PredictionServiceClient(client_options={"api_endpoint": location+"-aiplatform.googleapis.com"})

    def process(self, input_data):
        endpoint = input_data.get('endpoint')
        instances = input_data.get('instances')
        if not endpoint or not instances:
            raise ValueError("Both 'endpoint' and 'instances' are required.")
        
        # Construct the fully qualified endpoint.
        endpoint = self.client.endpoint_path(
            project=self.project, location=self.location, endpoint=endpoint
        )
        
        # The AI Platform services require the parameters to be formatted as a dictionary
        # The exact structure depends on the model
        parameters_dict = {}
        parameters = aiplatform.PredictRequest.Parameters(parameters_dict)
        
        response = self.client.predict(
            endpoint=endpoint,
            instances=instances,
            parameters=parameters
        )
        return response

    def post_process(self, result):
        # Depending on your requirements, you may need to extract the results in a different way
        return [prediction for prediction in result.predictions]

    def destroy(self):
        pass
