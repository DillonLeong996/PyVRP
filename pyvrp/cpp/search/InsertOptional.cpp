#include "InsertOptional.h"
#include "primitives.h"

using pyvrp::search::InsertOptional;

void InsertOptional::operator()(PerturbationContext const &context)
{
    if (context.numPerturbations == 0 || data_.numClients() == 0)
        return;

    auto const &costEvaluator = context.costEvaluator;
    size_t numInserts = 0;

    for (auto const uClient : context.orderNodes)
    {
        ProblemData::Client clientData = data_.location(uClient);
        if (clientData.required)
            continue;

        auto *U = &context.nodes[uClient];
        if (U->route())
            continue;

        auto [UAfter, cost] = bestInsert(U,
                                         data_,
                                         costEvaluator,
                                         context.neighbours,
                                         context.nodes,
                                         context.routes);

        UAfter->route()->insert(UAfter->idx() + 1, U);
        UAfter->route()->update();

        context.promising[U->client()] = true;
        context.promising[p(U)->client()] = true;
        context.promising[n(U)->client()] = true;

        if (++numInserts >= context.numPerturbations)
            break;
    }
}

InsertOptional::InsertOptional(ProblemData const &data) : data_(data) {}

template <>
bool pyvrp::search::supports<InsertOptional>(ProblemData const &data)
{
    // Only support InsertOptional if the problem has optional clients.
    auto const pred = [](auto const &client) { return !client.required; };
    return std::any_of(data.clients().begin(), data.clients().end(), pred);
}
