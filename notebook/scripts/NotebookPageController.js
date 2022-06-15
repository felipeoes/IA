import dataset from "../data/dataset.js";
import InputPageBuilder from "./InputPageBuilder.js";

import { sendMessage } from "./utils/web-socket-utils.js";
import { fetchFile } from "./utils/http-utils.js";

class NotebookPageController {
  static letters = ["A", "B", "C", "D", "E", "J", "K"];

  constructor() {
    this.letterInputs = document.querySelectorAll(".letter input");
    this.modal = document.querySelector("dialog");
    this.btn = document.getElementById("socket-send");

    this.attachEventListeners();
    this.buildTrainmentResultChart();
    this.buildTrainmentLogTable();
    this.buildInputs();
    this.buildTabIndexes();
  }

  buildTabIndexes() {
    document
      .querySelectorAll("main > section")
      .forEach((section, index) => section.setAttribute("tabindex", index + 1));
  }

  async buildTrainmentLogTable() {
    const html = await fetchFile("log_mlp_recorte.html");
    const container = document.querySelector(".log-table");

    container.insertAdjacentHTML("beforeend", html);
  }

  async fetchTrainmentResult() {
    const trainment = await fetchFile("MLP_Error.csv");
    const labels = [];
    const errors = [];

    trainment.split("\n").map((line) => {
      const [epoch, error] = line.split(",");

      labels.push(epoch);
      errors.push(parseFloat(error));
    });

    return [labels, errors];
  }

  async buildTrainmentResultChart() {
    const [labels, data] = await this.fetchTrainmentResult();

    const context = document.getElementById("mlp-error").getContext("2d");

    new Chart(context, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "Erro",
            data,
            backgroundColor: "#c21947",
            borderColor: "#e44c75",
          },
        ],
      },
      options: {},
    });
  }

  createPredictedTable(data) {
    const table = this.modal.querySelector("tbody");
    table.textContent = "";
    data.forEach(({ index, real }) => {
      table.insertAdjacentHTML(
        "beforeend",
        `
                <tr>
                    <td> ${index} </td>
                    <td> ${real.toFixed(4)} </td>
                </tr>
            `
      );
    });
  }

  sendMessage() {
    const predicted = this.modal.querySelector("p");
    const letters = NotebookPageController.letters;
    const parse = (input) => (input.checked ? 1 : -1);
    const step = (value) => (value + 0.05 > 1 ? 1 : 0);
    const letter = Array.from(
      document.querySelectorAll(".letter .letter-input")
    );
    const message = letter.slice(0, 63).map(parse);

    sendMessage({ values: Array.of(message) }, async (response) => {
      const [data] = await JSON.parse(response.data);

      const dataCopy = [...data];
      // pega os dois maiores valores
      const selected = dataCopy.sort((a, b) => b - a).slice(0, 2);

      // verifica se a distancia entre os dois maiores valores é menor que 0.2
      const isClose = selected[0] - selected[1] < 0.2;

      // pega o indice dos valores
      const index = data.indexOf(selected[0]);
      const index2 = data.indexOf(selected[1]);

      // se a distancia for menor que 0.2, então o valor predito é aceitável
      predicted.textContent = !isClose
        ? `Letra ${letters[index]}`
        : `Houve uma confusão entre: ${letters[index]} e ${letters[index2]}`;

      const formatted_data = data.map((element, index) => ({
        real: element,
        index,
        formatted: step(element),
      }));

      // const possible_letters = formatted_data.filter(
      //   (data) => data.formatted === 1
      // );

      // switch (possible_letters.length) {
      //   case 1:
      //     const letter = possible_letters.at(0);
      //     predicted.textContent = `Letra ${letters[letter.index]}`;
      //     break;
      //   case 2:
      //     const predictedLetters = possible_letters
      //       .map((data) => letters[data.index])
      //       .join(", ");
      //     predicted.textContent = `Houve uma confusão entre: ${predictedLetters}`;
      //     break;
      //   default:
      //     predicted.textContent = "Não consegui predizer sua letra";
      // }

      this.createPredictedTable(formatted_data);
      this.modal.showModal();
    });
  }

  getLetterInputs() {
    if (this.letterInputs.length === 0) {
      this.letterInputs = document.querySelectorAll(".letter input");
    }

    return this.letterInputs;
  }

  changeInputLetter(event) {
    const id = event.target.getAttribute("data-id");
    const data = dataset.at(id);
    this.getLetterInputs().forEach(
      (input, index) => (input.checked = data[index] === 1)
    );
  }

  buildInputs() {
    const nav = document.querySelector(".nav-letters");

    const createButton = (index) => {
      const button = document.createElement("button");
      button.setAttribute("type", "button");
      button.setAttribute("data-id", index);
      button.addEventListener("click", this.changeInputLetter.bind(this));
      button.textContent = `Letra ${NotebookPageController.letters[index]}`;
      return button;
    };

    Array.from({ length: dataset.length }, (_, index) => {
      nav.insertAdjacentElement("beforeend", createButton(index));
    });

    new InputPageBuilder({
      root: document.querySelector(".letter"),
      dataset: Array.of(dataset.at(0)),
      numberOfLettersPerRow: 7,
    });
  }

  attachEventListeners() {
    this.modal
      .querySelector("button")
      .addEventListener("click", () => this.modal.close());
    this.btn.addEventListener("click", this.sendMessage.bind(this));
  }
}

new NotebookPageController();
