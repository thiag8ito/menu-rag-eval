"""Gera cardápios sintéticos de forma determinística.

Decisão de design importante: os MESMOS pratos aparecem em vários restaurantes
(com preços diferentes). Isso cria "hard negatives" naturais: a pergunta
"Quanto custa a Margherita no Forno Antigo?" só é respondida corretamente se o
retriever distinguir o restaurante, não apenas o prato.
"""
import argparse
import json
import random
from pathlib import Path

NEIGHBORHOODS = ["Centro", "Jardim Paulista", "Vila Nova", "Alto da Boa Vista",
                 "Bela Vista", "Santa Cruz", "Jardim América", "Cidade Jardim",
                 "Vila Mariana", "Boa Esperança"]
CITIES = ["São Paulo", "Marília", "Campinas", "Ribeirão Preto", "Bauru"]
V, VG, SG, SL, PC, FIT = "vegetariano", "vegano", "sem glúten", "sem lactose", "picante", "fit"

# (nome, categoria, descrição, preço_base, tags)
CUISINES = {
    "Pizzaria": (["Forno", "Massa Fina", "Cantina", "Pizza"], [
        ("Pizza Margherita", "Pizzas", "molho de tomate, muçarela de búfala e manjericão fresco", 42.0, [V]),
        ("Pizza Calabresa", "Pizzas", "calabresa fatiada, cebola roxa e azeitonas pretas", 45.0, []),
        ("Pizza Quatro Queijos", "Pizzas", "muçarela, provolone, gorgonzola e parmesão", 52.0, [V]),
        ("Pizza Portuguesa", "Pizzas", "presunto, ovo, cebola, ervilha e azeitona", 48.0, []),
        ("Pizza Vegana de Cogumelos", "Pizzas", "queijo de castanha, shiitake, shimeji e rúcula", 54.0, [VG, V, SL]),
        ("Pizza de Frango com Catupiry", "Pizzas", "frango desfiado temperado e catupiry original", 49.0, []),
        ("Pizza Pepperoni Picante", "Pizzas", "pepperoni, pimenta calabresa e mel picante", 53.0, [PC]),
        ("Calzone de Presunto e Queijo", "Calzones", "massa recheada com presunto, muçarela e orégano", 39.0, []),
        ("Pizza Doce de Chocolate com Morango", "Pizzas Doces", "ganache de chocolate meio amargo e morangos frescos", 46.0, [V]),
        ("Pizza Napolitana", "Pizzas", "tomate fatiado, muçarela, parmesão e alho frito", 47.0, [V]),
        ("Refrigerante Lata 350ml", "Bebidas", "guaraná, cola ou limão", 6.5, [VG]),
        ("Suco de Laranja 500ml", "Bebidas", "suco natural sem adição de açúcar", 11.0, [VG, SG, SL]),
    ]),
    "Japonesa": (["Sakura", "Yamato", "Kinoko", "Tokyo"], [
        ("Combo 20 Peças", "Combos", "8 sashimis, 6 uramakis salmão e 6 hossomakis de pepino", 79.0, [SL]),
        ("Temaki de Salmão", "Temakis", "salmão fresco, cream cheese e cebolinha em alga crocante", 32.0, []),
        ("Temaki Vegano de Manga", "Temakis", "manga, pepino, cenoura e shitake na alga nori", 28.0, [VG, V, SG, SL]),
        ("Hot Roll", "Quentes", "uramaki empanado e frito recheado com salmão e cream cheese", 36.0, []),
        ("Yakisoba de Legumes", "Pratos Quentes", "macarrão oriental salteado com brócolis, cenoura e repolho", 38.0, [VG, V]),
        ("Yakisoba de Frango", "Pratos Quentes", "macarrão oriental com tiras de frango e legumes", 41.0, []),
        ("Sashimi de Salmão 10 Fatias", "Sashimis", "fatias de salmão fresco servidas com wasabi e gengibre", 44.0, [SG, SL]),
        ("Gyoza de Carne", "Entradas", "pastel japonês grelhado recheado com carne suína", 27.0, []),
        ("Missoshiru", "Entradas", "sopa de missô com tofu e cebolinha", 14.0, [VG, V]),
        ("Poke de Atum", "Pokes", "arroz japonês, atum, edamame, manga e molho shoyu", 46.0, [SL]),
        ("Uramaki Picante de Salmão", "Uramakis", "salmão com molho spicy de pimenta sriracha", 35.0, [PC]),
        ("Mochi de Morango", "Sobremesas", "bolinho de arroz recheado com sorvete de morango", 18.0, [V]),
    ]),
    "Hamburgueria": (["Burger", "Smash", "Brasa", "Grill"], [
        ("Smash Burger Duplo", "Hambúrgueres", "dois blends de 80g, queijo cheddar, picles e molho da casa", 36.0, []),
        ("Cheeseburger Clássico", "Hambúrgueres", "blend 160g, queijo prato, alface, tomate e maionese", 32.0, []),
        ("Burger Vegano de Grão-de-Bico", "Hambúrgueres", "hambúrguer de grão-de-bico, rúcula, tomate e maionese vegana", 34.0, [VG, V, SL]),
        ("Burger de Frango Crocante", "Hambúrgueres", "filé de frango empanado, alface e maionese de ervas", 31.0, []),
        ("Burger Bacon Picante", "Hambúrgueres", "blend 160g, bacon crocante, jalapeño e cheddar", 39.0, [PC]),
        ("Batata Frita Média", "Acompanhamentos", "batatas fritas com sal e páprica defumada", 18.0, [VG, V, SG]),
        ("Onion Rings", "Acompanhamentos", "anéis de cebola empanados e fritos", 20.0, [V]),
        ("Nuggets 10 Unidades", "Acompanhamentos", "nuggets de frango com molho barbecue", 24.0, []),
        ("Milkshake de Chocolate 400ml", "Sobremesas", "sorvete de creme, calda de chocolate e chantilly", 22.0, [V]),
        ("Salada Caesar Fit", "Saladas", "alface americana, frango grelhado, parmesão e croutons integrais", 29.0, [FIT]),
        ("Brownie com Sorvete", "Sobremesas", "brownie quente com bola de sorvete de baunilha", 19.0, [V]),
        ("Refrigerante Lata 350ml", "Bebidas", "guaraná, cola ou limão", 6.5, [VG]),
    ]),
    "Brasileira": (["Marmita", "Sabor da Roça", "Cozinha", "Tempero"], [
        ("Feijoada Completa", "Pratos do Dia", "feijão preto com carnes suínas, arroz, couve, farofa e laranja", 38.0, [SL]),
        ("Prato Feito de Frango Grelhado", "Marmitex", "filé de frango, arroz, feijão, salada e batata sauté", 26.0, [SL]),
        ("Strogonoff de Carne", "Marmitex", "strogonoff cremoso, arroz branco e batata palha", 29.0, []),
        ("Marmita Fit de Patinho", "Marmitex", "patinho moído, arroz integral, brócolis e cenoura cozida", 27.0, [FIT, SL]),
        ("Moqueca de Peixe", "Pratos do Dia", "peixe em leite de coco e dendê, arroz e pirão", 45.0, [SL]),
        ("Virado à Paulista", "Pratos do Dia", "tutu de feijão, bisteca, couve, ovo, banana e arroz", 34.0, [SL]),
        ("Feijão Tropeiro", "Acompanhamentos", "feijão, farinha de mandioca, bacon, linguiça e ovo", 24.0, []),
        ("Parmegiana de Frango", "Marmitex", "frango à parmegiana com arroz e batatas fritas", 31.0, []),
        ("Prato Vegano de Lentilha", "Marmitex", "lentilha, arroz, legumes assados e farofa de castanhas", 25.0, [VG, V, SL]),
        ("Pudim de Leite", "Sobremesas", "pudim cremoso de leite condensado com calda de caramelo", 12.0, [V, SG]),
        ("Suco de Maracujá 500ml", "Bebidas", "suco natural de maracujá", 10.0, [VG, SG, SL]),
        ("Escondidinho de Carne Seca", "Marmitex", "purê de mandioca gratinado recheado com carne seca", 33.0, []),
    ]),
    "Vegana": (["Raiz", "Verde", "Folha", "Semente"], [
        ("Bowl de Quinoa e Grão-de-Bico", "Bowls", "quinoa, grão-de-bico assado, abobrinha e molho tahine", 33.0, [VG, V, SG, SL, FIT]),
        ("Hambúrguer de Cogumelos", "Lanches", "burger de shiitake e castanhas no pão integral", 31.0, [VG, V, SL]),
        ("Lasanha de Berinjela", "Pratos", "berinjela grelhada, molho de tomate e queijo de castanha", 36.0, [VG, V, SG]),
        ("Curry de Legumes com Coco", "Pratos", "legumes ao curry em leite de coco com arroz basmati", 35.0, [VG, V, SG, SL, PC]),
        ("Salada de Folhas com Tofu", "Saladas", "mix de folhas, tofu grelhado, tomate cereja e gergelim", 28.0, [VG, V, SG, FIT]),
        ("Wrap de Falafel", "Lanches", "falafel, homus, picles de cebola roxa e folhas", 27.0, [VG, V, SL]),
        ("Brigadeiro Vegano", "Sobremesas", "brigadeiro de cacau com leite de coco", 8.0, [VG, V, SG, SL]),
        ("Suco Verde Detox 400ml", "Bebidas", "couve, maçã, limão e gengibre", 14.0, [VG, SG, SL, FIT]),
        ("Panqueca de Espinafre", "Pratos", "panqueca recheada com espinafre e ricota de tofu", 32.0, [VG, V, SL]),
        ("Feijoada Vegana", "Pratos", "feijão preto com cogumelos defumados, arroz e couve", 34.0, [VG, V, SG, SL]),
        ("Bolo de Cenoura Vegano", "Sobremesas", "bolo de cenoura com cobertura de chocolate sem leite", 13.0, [VG, V, SL]),
        ("Açaí Bowl com Granola", "Bowls", "açaí batido com banana, granola e frutas vermelhas", 26.0, [VG, V, SL]),
    ]),
    "Açaí e Sobremesas": (["Açaí", "Doce", "Gelato", "Sorvete"], [
        ("Açaí 500ml com 3 Acompanhamentos", "Açaís", "açaí cremoso com granola, banana e leite em pó", 24.0, [V]),
        ("Açaí Zero Fit 400ml", "Açaís", "açaí sem açúcar com whey, pasta de amendoim e frutas", 28.0, [FIT, SL]),
        ("Sorvete de Pistache", "Sorvetes", "sorvete artesanal de pistache com calda de chocolate branco", 21.0, [V, SG]),
        ("Petit Gâteau", "Sobremesas", "bolinho quente de chocolate com sorvete de creme", 26.0, [V]),
        ("Cheesecake de Frutas Vermelhas", "Sobremesas", "torta gelada com calda de frutas vermelhas", 23.0, [V]),
        ("Milkshake de Morango 400ml", "Milkshakes", "sorvete de morango e calda de frutas", 19.0, [V, SG]),
        ("Picolé de Frutas Veganas", "Sorvetes", "picolé de manga, limão ou abacaxi sem leite", 9.0, [VG, V, SG, SL]),
        ("Brownie Vegano", "Sobremesas", "brownie de cacau 70% sem ovos e sem leite", 14.0, [VG, V, SL]),
        ("Crepe de Nutella com Banana", "Crepes", "crepe doce com creme de avelã e banana", 22.0, [V]),
        ("Waffle com Sorvete", "Waffles", "waffle belga, sorvete de creme e calda de caramelo", 25.0, [V]),
        ("Cookie Recheado", "Sobremesas", "cookie de baunilha recheado com doce de leite", 12.0, [V]),
        ("Água de Coco 300ml", "Bebidas", "água de coco gelada", 7.0, [VG, SG, SL]),
    ]),
    "Árabe": (["Tenda", "Oásis", "Beirute", "Cedro"], [
        ("Esfiha de Carne", "Esfihas", "massa macia com carne moída temperada e limão", 6.5, [SL]),
        ("Esfiha de Queijo", "Esfihas", "massa macia recheada com muçarela cremosa", 6.0, [V]),
        ("Kibe Frito", "Salgados", "bolinho de trigo e carne bovina com hortelã", 8.0, [SL]),
        ("Kafta no Pão Sírio", "Sanduíches", "espetinho de kafta, tomate, cebola e molho de iogurte", 29.0, []),
        ("Homus com Pão Sírio", "Entradas", "pasta de grão-de-bico com tahine e azeite", 22.0, [VG, V, SL]),
        ("Falafel Prato", "Pratos", "6 bolinhos de grão-de-bico, salada e molho tahine", 31.0, [VG, V, SL]),
        ("Shawarma de Frango", "Sanduíches", "frango marinado, picles, alho e pão sírio", 28.0, [SL]),
        ("Arroz com Lentilha (Mujaddara)", "Pratos", "arroz e lentilhas com cebola caramelizada", 26.0, [VG, V, SG, SL]),
        ("Baklava 3 Unidades", "Sobremesas", "massa folhada, nozes e calda de mel", 18.0, [V]),
        ("Tabule", "Saladas", "trigo, salsinha, tomate, hortelã e limão", 20.0, [VG, V, SL]),
        ("Esfiha Picante de Frango", "Esfihas", "frango desfiado com pimenta síria", 7.0, [PC, SL]),
        ("Chá de Hortelã", "Bebidas", "chá quente de hortelã fresca", 8.0, [VG, SG, SL]),
    ]),
    "Massas": (["Nonna", "Cantina", "Al Dente", "Pasta"], [
        ("Espaguete à Carbonara", "Massas", "espaguete, bacon, gema e queijo pecorino", 42.0, []),
        ("Fettuccine ao Alfredo", "Massas", "fettuccine em molho cremoso de queijo parmesão", 40.0, [V]),
        ("Penne ao Sugo Vegano", "Massas", "penne com molho de tomate italiano e manjericão", 34.0, [VG, V, SL]),
        ("Lasanha à Bolonhesa", "Massas", "camadas de massa, molho bolonhesa e queijo gratinado", 44.0, []),
        ("Ravioli de Queijo com Nozes", "Massas", "ravioli recheado com ricota e nozes ao molho de sálvia", 46.0, [V]),
        ("Gnocchi ao Pesto", "Massas", "nhoque de batata com molho pesto de manjericão", 39.0, [V]),
        ("Penne Arrabbiata", "Massas", "penne com molho de tomate e pimenta calabresa", 35.0, [VG, V, PC, SL]),
        ("Risoto de Cogumelos", "Risotos", "arroz arbóreo com mix de cogumelos e parmesão", 48.0, [V, SG]),
        ("Macarrão sem Glúten ao Alho e Óleo", "Massas", "massa de arroz com alho dourado e azeite", 36.0, [VG, V, SG, SL]),
        ("Tiramisù", "Sobremesas", "creme de mascarpone, café e cacau", 20.0, [V]),
        ("Salada Caprese", "Entradas", "tomate, muçarela de búfala e manjericão", 27.0, [V, SG]),
        ("Pão de Alho", "Entradas", "pão italiano com manteiga de alho e ervas", 14.0, [V]),
    ]),
}


REVIEW_TEMPLATES = [
    "{dish}: chegou no prazo e bem embalado. Recomendo!",
    "Gostei de pedir {dish}, mas a embalagem poderia ser melhor.",
    "Atendimento excelente e preço justo. Sempre peço {dish}.",
    "A entrega demorou um pouco, mas {dish} compensou.",
    "Ótimo custo-benefício. Voltarei a pedir {dish}.",
    "Quanto ao valor, achei razoável; {dish} é o meu preferido.",
    "Pedi {dish} para a família toda e todo mundo aprovou.",
    "O cardápio é variado; {dish} é o destaque da casa.",
]


def money(x: float) -> str:
    return f"{x:.2f}".replace(".", ",")


def render_item_line(name, price, description, tags):
    tag_txt = f" Opções: {', '.join(tags)}." if tags else ""
    return f"- {name} (R$ {money(price)}): {description}.{tag_txt}"


def build_restaurant(rid: int, cuisine: str, rng: random.Random, used_names: set):
    prefixes, pool = CUISINES[cuisine]
    suffixes = ["do Zé", "da Praça", "Premium", "Express", "Gourmet", "da Vila", "Brasil",
                "Central", "da Esquina", "Família", "do Bairro", "Real", "Nobre", "Fresh"]
    while True:
        name = f"{rng.choice(prefixes)} {rng.choice(suffixes)}"
        if name not in used_names:
            used_names.add(name)
            break
    n_items = rng.randint(8, len(pool))
    chosen = rng.sample(pool, n_items)
    items = []
    for j, (dname, cat, desc, base, tags) in enumerate(chosen):
        price = round(base * rng.uniform(0.85, 1.30) - 0.1, 1) + 0.0
        price = max(price, 3.9)
        items.append({
            "dish_id": f"r{rid:03d}-d{j:02d}", "name": dname, "category": cat,
            "description": desc, "price": round(price, 2), "tags": tags,
            "line": render_item_line(dname, round(price, 2), desc, tags),
        })
    items.sort(key=lambda x: x["category"])
    doc = {
        "restaurant_id": f"r{rid:03d}", "name": name, "cuisine": cuisine,
        "neighborhood": rng.choice(NEIGHBORHOODS), "city": rng.choice(CITIES),
        "delivery_min": (dm := rng.choice([20, 25, 30, 40])), "delivery_max": dm + 15,
        "delivery_fee": round(rng.choice([0, 3.9, 5.9, 7.9]), 2),
        "items": items,
    }
    lines = [
        f"# {doc['name']}",
        f"Culinária: {cuisine} | Bairro: {doc['neighborhood']}, {doc['city']} | "
        f"Entrega: {doc['delivery_min']}-{doc['delivery_max']} min | Taxa de entrega: R$ {money(doc['delivery_fee'])}",
    ]
    # seção "Sobre": texto corrido, sem preços nem ingredientes (ruído realista no topo do documento)
    year = rng.randint(2005, 2022)
    lines += ["## Sobre",
              f"O {doc['name']} é uma casa de culinária {cuisine.lower()} no bairro {doc['neighborhood']}, em "
              f"{doc['city']}, aberta desde {year}. Trabalhamos com ingredientes frescos e preparo diário."]
    cur = None
    for it in items:
        if it["category"] != cur:
            cur = it["category"]
            lines.append(f"## {cur}")
        lines.append(it["line"])
    # avaliações: citam pratos do MESMO restaurante, mas nunca ingredientes/preços exatos.
    # Funcionam como distratores intra-documento (mesmo nome de prato, sem a resposta).
    lines.append("## Avaliações de clientes")
    for tpl in rng.sample(REVIEW_TEMPLATES, k=rng.randint(4, 6)):
        stars = rng.choice([3, 4, 4, 5, 5])
        lines.append(f"{'★' * stars}{'☆' * (5 - stars)} \"{tpl.format(dish=rng.choice(items)['name'])}\"")
    doc["text"] = "\n".join(lines)
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/menus.jsonl")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    cuisines = list(CUISINES)
    used: set = set()
    docs = [build_restaurant(i, cuisines[i % len(cuisines)], rng, used) for i in range(a.n)]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    n_items = sum(len(d["items"]) for d in docs)
    print(f"{len(docs)} restaurantes, {n_items} pratos -> {a.out}")


if __name__ == "__main__":
    main()
