import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# Definições de cores e fontes para o estilo da interface
BG_COLOR = "#0A192F"  # Cor de fundo da janela
ENTRY_BG = "#112240"  # Cor de fundo dos campos de entrada
BUTTON_BG = "#64FFDA"  # Cor de fundo do botão
TEXT_COLOR = "#FFFFFF"  # Cor do texto
FONT = ("Segoe UI", 12)  # Fonte do texto

def classificar_imc(imc):
    """
    Função para classificar o IMC e retornar uma descrição do estado de saúde baseado no valor do IMC.
    
    Parâmetros:
        imc (float): O valor calculado do IMC.

    Retorna:
        (str): Descrição do estado de saúde baseado no IMC.
    """
    if imc < 18.5:
        return (
            "IMC < 18,5 kg/m²: Baixo peso.\n"
            "É recomendado procurar um médico para avaliação criteriosa do resultado. "
            "Pode indicar um estado de consumo do organismo, com poucas reservas e riscos associados."
        )
    elif 18.5 <= imc <= 24.9:
        return (
            "IMC entre 18,5 e 24,9 kg/m²: Peso adequado.\n"
            "Tudo indica que está tudo bem, mas é importante avaliar outros parâmetros da composição corporal, "
            "para compreender se estão dentro do recomendado. Algumas pessoas apresentam IMC dentro da normalidade, "
            "mas têm circunferência abdominal maior que a recomendada e/ou quantidade de massa gorda acima do ideal."
        )
    elif 25 <= imc <= 29.9:
        return (
            "IMC entre 25 e 29,9 kg/m²: Sobrepeso.\n"
            "O sobrepeso está associado ao risco de doenças como diabetes e hipertensão. Então, atenção! "
            "Consulte um médico e reveja hábitos para reverter o quadro. Também é importante avaliar outros parâmetros, "
            "como a circunferência abdominal."
        )
    elif 30 <= imc <= 34.9:
        return (
            "IMC entre 30,0 e 34,9 kg/m²: Obesidade grau I.\n"
            "É importante buscar orientação médica e nutricional para entender melhor o seu caso, mesmo que os exames "
            "(colesterol e glicemia, por exemplo) estejam normais."
        )
    elif 35 <= imc <= 39.9:
        return (
            "IMC entre 35,0 e 39,9 kg/m²: Obesidade grau II.\n"
            "Indica um quadro de obesidade mais evoluído em relação à classificação anterior e, mesmo com exames laboratoriais "
            "dentro da normalidade, não se deve atrasar a busca por orientação médica e nutricional."
        )
    else:
        return (
            "IMC ≥ 40,0 kg/m²: Obesidade grau III.\n"
            "Nesse ponto, a chance de já estarmos diante de outras doenças associadas é mais elevada. "
            "É fundamental buscar orientação médica."
        )

def calcular_imc():
    """
    Função para calcular o IMC com base nos valores inseridos de peso e altura, 
    e exibir a classificação do IMC.
    """
    try:
        # Obtém e formata as entradas de peso e altura
        peso_str = entry_peso.get().replace(",", ".")
        altura_str = entry_altura.get().replace(",", ".")
        
        # Verifica se as entradas são números válidos
        if not peso_str.replace(".", "", 1).isdigit() or not altura_str.replace(".", "", 1).isdigit():
            raise ValueError("A entrada deve ser um número válido.")
        
        peso = float(peso_str)
        altura = float(altura_str)

        # Normaliza a altura caso esteja em centímetros
        if altura > 3:
            altura = altura / 100  

        # Valida a altura e o peso
        if altura < 0.5 or altura > 2.5:
            raise ValueError("Altura em metros fora do intervalo esperado.")
        if peso <= 0:
            raise ValueError("Peso deve ser maior que zero.")

        # Calcula o IMC e obtém a classificação
        imc = peso / (altura ** 2)
        classificacao = classificar_imc(imc)

        # Atualiza a label com o resultado
        label_resultado.config(
            text=f"Altura considerada: {altura:.2f} m\nIMC: {imc:.2f}\n\n{classificacao}",
            fg=TEXT_COLOR
        )

    except ValueError as e:
        # Exibe uma mensagem de erro em caso de entradas inválidas
        messagebox.showerror("Erro", f"Entrada inválida: {e}")

def on_focus_in(event, entry, placeholder):
    """
    Função para remover o placeholder quando o campo de entrada recebe foco.
    """
    if entry.get() == placeholder:
        entry.delete(0, tk.END)
        entry.config(fg=TEXT_COLOR)

def on_focus_out(event, entry, placeholder):
    """
    Função para restaurar o placeholder quando o campo de entrada perde o foco.
    
    """
    if entry.get() == "":
        entry.insert(0, placeholder)
        entry.config(fg="#888888")  

# Janela principal
root = tk.Tk()
root.title("Calculadora de IMC")
root.geometry("800x800")  # Define o tamanho da janela
root.configure(bg=BG_COLOR)

# Carrega a imagem e a redimensiona
image = Image.open("fit.png")
image = image.resize((500, 300), Image.Resampling.LANCZOS)
img_tk = ImageTk.PhotoImage(image)

# Exibe a imagem na parte superior da janela
label_img = tk.Label(root, image=img_tk, bg=BG_COLOR)
label_img.image = img_tk
label_img.pack(pady=20)

# Título da aplicação
label_titulo = tk.Label(root, text="Calculadora de IMC", font=("Segoe UI", 16, "bold"), fg=TEXT_COLOR, bg=BG_COLOR)
label_titulo.pack(pady=10)

# Texto do placeholder
peso_placeholder = "Digite seu peso"
altura_placeholder = "Digite sua altura"

# Campo de entrada de peso
frame_peso = tk.Frame(root, bg=BG_COLOR)
frame_peso.pack(pady=5, padx=20, fill=tk.X)
tk.Label(frame_peso, text="Peso (kg):", font=FONT, fg=TEXT_COLOR, bg=BG_COLOR).pack(side=tk.LEFT, padx=5)
entry_peso = tk.Entry(frame_peso, font=FONT, bg=ENTRY_BG, fg="#888888", insertbackground=TEXT_COLOR, bd=2, relief="solid")
entry_peso.insert(0, peso_placeholder)
entry_peso.bind("<FocusIn>", lambda event: on_focus_in(event, entry_peso, peso_placeholder))
entry_peso.bind("<FocusOut>", lambda event: on_focus_out(event, entry_peso, peso_placeholder))
entry_peso.pack(side=tk.LEFT, fill=tk.X, expand=True)
entry_peso.focus_set()

# Campo de entrada de altura
frame_altura = tk.Frame(root, bg=BG_COLOR)
frame_altura.pack(pady=5, padx=20, fill=tk.X)
tk.Label(frame_altura, text="Altura (m ou cm):", font=FONT, fg=TEXT_COLOR, bg=BG_COLOR).pack(side=tk.LEFT, padx=5)
entry_altura = tk.Entry(frame_altura, font=FONT, bg=ENTRY_BG, fg="#888888", insertbackground=TEXT_COLOR, bd=2, relief="solid")
entry_altura.insert(0, altura_placeholder)
entry_altura.bind("<FocusIn>", lambda event: on_focus_in(event, entry_altura, altura_placeholder))
entry_altura.bind("<FocusOut>", lambda event: on_focus_out(event, entry_altura, altura_placeholder))
entry_altura.pack(side=tk.LEFT, fill=tk.X, expand=True)

# Botão para calcular IMC
btn_calcular = tk.Button(root, text="Calcular IMC", font=FONT, bg=BUTTON_BG, fg=BG_COLOR, command=calcular_imc, bd=2, relief="solid")
btn_calcular.pack(pady=20, fill=tk.X, padx=20)

# Label para exibir o resultado
label_resultado = tk.Label(root, text="", font=("Segoe UI", 12), fg=TEXT_COLOR, bg=BG_COLOR, wraplength=450, justify="left")
label_resultado.pack(padx=20, pady=10)

# Tecla Enter para calcular
root.bind("<Return>", lambda event: calcular_imc())

# Inicia o loop da interface gráfica
root.mainloop()
