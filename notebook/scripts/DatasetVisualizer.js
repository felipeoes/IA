import InputPageBuilder from './InputPageBuilder.js';

const or = (value, condition, other) => (value === condition) ? value : other

const parseElement = (value) => or(parseInt(value), 1, -1)

const parseLines = (data) => data.split("\n")

const parseLine = (line) => line.split(",").map(parseElement)

const pipe = (...fns) => (data) => fns.reduce( (actual, fn) => fn(actual), data)

const chunks = (array, size = 21) => array.reduce( (chunks, element, index) => {
    if (index % size === 0 ){
        chunks.push(Array.of(element))
    }
    else {
        chunks.at(-1).push(element)
    }

    return chunks
}, [])

const parse = pipe(
    parseLines,
    (lines) => lines.map(parseLine),
)

const createPage = (chunks) => {
    const containers = document.querySelectorAll(".letters-container")

    chunks.forEach( (dataset, index) => {
        new InputPageBuilder({
            root: containers[index],
            btn: document.getElementById("btn-download"),
            dataset: dataset,
            numberOfLettersPerRow: 7
        })
    })
}

export default (file) => fetch(file)
                        .then( data => data.text() )
                        .then(parse)
                        .then( dataset => chunks(dataset) )
                        .then(createPage)
                        .catch(console.error)