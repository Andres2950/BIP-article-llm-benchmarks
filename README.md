# BIP-article-llm-benchmarks
A python program to do benchmarks on LLM models that run in Ollama


## Ollama

For this program to work you need Ollama to run local models.
The following instructions will help you install and setup Ollama in Arch Linux.

### Instaling ollama

First installl Ollama:

```
$ pacman -S ollama
```

Start the Ollama services:

```
$ systemctl start ollama
$ systemctl enable ollama
```

### Downloading the model and test it

This example uses the model Llama3.2-3B, the model can be changed according to the computer.
For this I am using my laptop with 16 GB RAM and a AMD Ryzen 7 7730U with Radeon Graphics, the GPU is integrated, not dedicated.


You donwload the model like this:

```
$ ollama pull llama3.2:3b
```

You may confirm that the model downloaded like this:

```
$ ollama list
```

If you have it locally and ready to run it should show you an output like this:

```
NAME           ID              SIZE      MODIFIED       
llama3.2:3b    a80c4f17acd5    2.0 GB    29 seconds ago    
```


Run the model to test it on the terminal.
You can talk with it here, once you are ready you may exit the conversation with the command ```/bye```:

```
$ ollama run llama3.2:3b


>>> Hola
¡hola! ¿En qué puedo ayudarte hoy?

>>> Soy Óscar
¡Genial, Oscar! Me alegra conocerte. ¿Qué te trae aquí hoy? ¿Necesitas ayuda con algo en particular o solo quieres charlar un rato?

>>> Qué modelo eres?
Soy un modelo de lenguaje llamado "Llama". Llama es una abreviatura para "Large Language Model Meta AI", y se refiere a mi capacidad para entender y generar texto. Soy un modelo de inteligencia artificial 
entrenado en grandes cantidades de datos, lo que me permite responder preguntas y completar tareas de texto con precisión y confianza.

Soy una herramienta de aprendizaje automático desarrollada por Meta AI, y estoy aquí para ayudarte en cualquier cosa que necesites. ¿Quieres saber más sobre cómo funciono o qué puedo hacer por ti?

>>> Puedes escribir código? haz un hello world en C++
¡Claro que sí! Como modelo de lenguaje, puedo generar código en diferentes idiomas, incluyendo C++. Aquí te dejo un "Hello World" en C++:

` ` `cpp
#include <iostream>

int main() {
    std::cout << "Hola, mundo!" << std::endl;
    return 0;
}
` ` `

Este es un programa básico que imprime la frase "Hola, mundo!" al console. El código se divide en dos partes principales:

1. La línea `#include <iostream>` importa el archivo de entrada estándar (`<iostream>`) del C++, que nos permite utilizar la función `std::cout`.
2. La función `main()` es la función principal del programa, donde se ejecuta cuando el programa se inicia.
3. Dentro de la función `main()`, se utiliza `std::cout` para imprimir la frase "Hola, mundo!" al console y luego se devuelve 0 para indicar que el programa ha terminado correctamente.

Recuerda que este es solo un ejemplo básico, y en una aplicación real, probablemente querrás agregar más funcionalidades y estructuras.

¿Quieres ver algún otro código o tienes alguna pregunta sobre C++?

>>> //bye
```

You can see the models that are lodaded in RAM like this:

```
$ ollama ps
```

You should see an output similar to:

```
NAME           ID              SIZE      PROCESSOR    CONTEXT    UNTIL              
llama3.2:3b    a80c4f17acd5    2.6 GB    100% CPU     4096       4 minutes from now    
```

Ollama mantains the models in RAM a certain ammount of time (marked as UNTIL).
Each time a request is done to the model, the timer is reset.
If the timer runs out, Ollama unloads the model from RAM so that it does not overwhelm the PC.


### Models tested


| Model | RAM Usage | Response time | Response quality |
| --- | --- | --- | --- |
| llama3.2:3b | 2GB ~ 3GB | Between 0.1s and 20s | *Waiting to test* |
| qwen2.5-coder:3b | 1GB ~ 1.5GB | Between 0.1s and 20s | *Waiting to test* |