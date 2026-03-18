import ollama 
import pathlib
def main():
 
    img_directory=pathlib.Path("Modulo3_NewsScrapper/pruebaOllama/imgs")

    List=[path for path in img_directory.iterdir()]
               
    prompt = """
        Analiza esta imagen y evalúa su posible impacto en el precio del stock de NVIDIA.
        Devuelve únicamente un JSON con la siguiente estructura:

        {
        "score": número,      # positivo = alcista, negativo = bajista, 0 = neutro
        "analisis": "breve explicación justificando el score"
        }

        No añadas texto adicional ni comentarios.
        """


    for img in List:
        response = ollama.chat(
            model="llama3.2-vision",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [img]  # <--- aquí va dentro de una lista
                }
            ]
        )
        print(response["message"]["content"])

if __name__ == "__main__":
    main()
