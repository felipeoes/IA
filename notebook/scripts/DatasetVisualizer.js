import InputPageBuilder from './InputPageBuilder.js';

const or = (value, condition, other) => (value === condition) ? value : other

const parseElement = (value) => or(parseInt(value), 1, -1)

const parseLines = (data) => data.split("\n")

const removeEmptyLines = (lines) => lines.filter( line => line.trim() !== "")

const parseLine = (line) => line.split(",").map(parseElement)

const pipe = (...fns) => (data) => fns.reduce( (actual, fn) => fn(actual), data)

const chunks = (array, size = 30) => array.reduce( (chunks, element, index) => {
    if (index % size === 0 ){
        chunks.push(Array.of(element))
    }
    else {
        chunks.at(-1).push(element)
    }

    return chunks
}, [])

const log_size = (value) => {
    console.log(value.length)
    return value
}

const validador_de_chunk = (chunk, chunk_index, { expected_letter_length = 74, expected_target_length = 11} = {}) => {
    chunk.values.forEach( ({ letter_lenght, target }, letter_index) => {
        console.log(`Chunk[${chunk_index}] - Letra de índice ${letter_index}`)
        const shouldHave73Size = letter_lenght === expected_letter_length
        const shouldHaveSize10 = target.length === expected_target_length

        const formatted_target = target.map( (element, index) => ({ element, index }))

        const filtered_target = formatted_target.filter( target => target.element === 1)

        const shouldHaveOnlyOneOne = filtered_target.length === 1

        const indexOfActiveNeuron = target.indexOf(1)

        console.log(`
            Indice[${letter_index}] - Letra[${letter_lenght}] - Target[${target.length}]
            A letra tem 74 caracteres ? => ${shouldHave73Size}
            O target tem 10 caracteres ? => ${shouldHaveSize10}
            Existe apenas um neuronio ativo ? => ${shouldHaveOnlyOneOne}
            Indice do neurônio ativo => ${indexOfActiveNeuron}
        `)

        if(!shouldHaveOnlyOneOne){
            console.table(target)
            console.log(filtered_target)
        }

        console.log("=======================================")
    })
}

const log_properties = (chunks) => {
    
    const result = chunks.map( chunk => {
        
        const length = chunk.length

        const values = chunk.map( letter => {
            const letter_lenght = letter.length
            const representation = letter.slice(0, 63)
            const target = letter.slice(63)

            return {
                letter_lenght,
                representation_length: representation.length,
                target
            }
        })

        return {
            length,
            values
        }

    })

    result.forEach( (r, i) => validador_de_chunk(r, i))

    // console.log(result)

    return chunks
}

const parse = pipe(
    parseLines,
    log_size,
    removeEmptyLines,
    log_size,
    (lines) => lines.map(parseLine),
)

export const createPage = (chunks, numberOfLettersPerRow) => {
    const containers = document.querySelectorAll(".letters-container")
    console.log(containers)

    chunks.forEach( (dataset, index) => {
        new InputPageBuilder({
            root: containers[index],
            btn: document.getElementById("btn-download"),
            dataset: dataset,
            numberOfLettersPerRow

        })
    })
}

export default ({
    file = "./data/datasets/caracteres-limpo.csv",
    with_chunks = false,
    number_of_letters = 7,
    chunk_size = 11 
} = { }) => fetch(file)
                        .then( data => data.text() )
                        .then(parse)
                        .then( dataset => {
                            return (with_chunks)
                                ? chunks(dataset, chunk_size)
                                : Array.of(dataset)
                        })
                        .then(log_properties)
                        .then( (dataset) => createPage(dataset, number_of_letters))