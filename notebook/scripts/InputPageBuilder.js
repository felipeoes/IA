class InputPageBuilder {
  constructor({ root, dataset, numberOfLettersPerRow, btn }) {
    this.btn = btn;
    this.dataset = dataset;
    this.root = root;
    this.numberOfLettersPerRow = numberOfLettersPerRow;
    this.containers = this.buildLettersContainer();
    this.init();
  }

  init() {
    this.buildLetters();
    this.btn?.addEventListener("click", this.download.bind(this));
  }

  drawLetter(container, index, checked) {
    const template = `
            <label class="letter-pixel ${
              index >= 63 ? "--none" : ""
            }" data-index=${index}>
                <input class="letter-input" type="checkbox" ${
                  checked ? "checked" : ""
                } />
                <span  class="letter-decoration">&nbsp;</span>
            </label>
        `;

    container.insertAdjacentHTML("beforeend", template);
  }

  buildLetters() {
    for (let index = 0; index < this.dataset.length; index++) {
      const line = this.dataset[index];
      const container = this.containers[index];

      for (let column = 0; column < line.length; column++) {
        const pixel = line[column];
        const isChecked = pixel === 1;
        this.drawLetter(container, column, isChecked);
      }
    }
  }

  buildLettersContainer() {
    const containers = Array.from({ length: this.dataset.length }, (_) => {
      const container = document.createElement("div");
      container.classList.add("letter-container");
      this.root.appendChild(container);

      return container;
    });

    return containers;
  }

  download() {
    const inputs = Array.from(
      this.root.querySelectorAll("input[type='checkbox']")
    );
    const numberOfPixelPerLetter = inputs.length / this.dataset.length;

    const parsed = inputs.map((input) => (input.checked ? 1 : -1));

    const chunks = parsed.reduce((matrix, array, index) => {
      const chunkIndex = Math.trunc(index / numberOfPixelPerLetter);

      if (!matrix.at(chunkIndex)) {
        matrix[chunkIndex] = [];
      }

      matrix[chunkIndex].push(array);

      return matrix;
    }, []);

    const csv = chunks
      .map((chunk) => chunk.join(","))
      .reduce((acc, act) => acc.concat(act).concat("\n"), "");

    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);

    this.btn.href = url;
    this.btn.setAttribute("download", "caracteres_js.csv");
  }
}

export default InputPageBuilder;
// const page = new InputPageBuilder({
//     root: document.querySelector(".letters-container"),
//     btn: document.getElementById("btn-download"),
//     dataset: dataset_limpo,
//     numberOfLettersPerRow: 7
// })

// page.download()
