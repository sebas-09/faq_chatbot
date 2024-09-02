from locust import HttpUser, TaskSet, task, between

class UserBehavior(TaskSet):
    @task
    def ask_question(self):
        self.client.post("/chatbot", json={"query": "hola"})

class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    wait_time = between(1, 5)

if __name__ == "__main__":
    import os
    os.system("locust -f locustfile.py")
